import zlib
import struct
import uproot

from storage import DQMDataStore
from rendering import DQMRenderer

class DQMAPIService:

    def __init__(self):
        self.store = DQMDataStore()

    def get_samples(self, run, dataset):
        if run == '':
            run = None
        if dataset == '':
            dataset = None
        
        results = self.store.get_samples(run, dataset)

        # TODO: offline_data should probably be removed. No GUI flavor distinction is required.
        samples = {
            'samples': [{
                'type': 'offline_data',
                'items': [{
                    'run': row[0],
                    'dataset': row[1]
                } for row in results]
            }]
        }

        return samples

    def get_archive(self, run, dataset, path, search):

        def get_subitem(full_path, current_dir):
            """Returns a closest sub item of full_path inside current_dir
            The second value of the response tuple indicates if that closest sub item is final one
            i.e. if it is a file or a directory.
            If full_path is a/b/c/d/file and current_dir is /a/b/c function will return (d, False).
            If full_path is a/b/c/d/file and current_dir is /a/b/c/d function will return (file, True).
            If current_dir is not part of full_path, function will return False.
            """

            # Remove double slashes
            full_path = full_path.replace('//', '/')
            current_dir = current_dir.replace('//', '/')

            if full_path.startswith(current_dir):
                names = full_path.replace(current_dir, '').split('/')
                if len(names) == 1: # This is an ME
                    return (names[0], True)
                else: # This is a folder
                    return (names[0], False)
            else:
                return False

        # Path must end with a slash
        if path and not path.endswith('/'):
            path = path + '/'
        
        blob = self.store.get_me_list_blob(run, dataset)

        buf = zlib.decompress(blob)
        lines = buf.split(b'\n')

        regex = re.compile(search) if search else None
        dirs = set()
        objs = set()

        # When doing dir listing we don't care about the secondary string
        # TODO: This is now a linear search over a sorted list: optimize this!!!
        for x in range(0, len(lines), 2):
            line = lines[x].decode("utf-8")
            subitem = get_subitem(line, path)
            if subitem:
                if regex and not regex.match(line.split('/')[-1]):
                    continue # Regex is provided and ME name doesn't match it

                if subitem[1]:
                    objs.add(subitem[0])
                else:
                    dirs.add(subitem[0])

        # Remove last slash before returning
        path = path[:-1]

        # Put results to a list to be returned
        data = {'contents': []}
        data['contents'].extend({'subdir': x} for x in dirs)
        data['contents'].extend({'obj': x, 'dir': path} for x in objs)

        return data

    async def get_rendered_image(self, run, dataset, path, width=200, height=200):

        width = int(width)
        height = int(height)

        filename, list_blob, offsets_blob = self.store.get_blobs_and_filename(run, dataset)
        list_buf = zlib.decompress(list_blob)
        list_lines = list_buf.split(b'\n')
        offsets_buf = zlib.decompress(offsets_blob)

        # Find the index of run/dataset/me in me list blob. 
        # The index in me list will correspond to the index in offsets list.
        # TODO: use binary search or smth!!!
        index = -1
        for i in range(0, len(list_lines), 2):
            line = list_lines[i]
            # For now replace double slash. This should not occur normally.
            if line.decode("utf-8").replace('//', '/') == path:
                index = i
                break
        
        if index == -1:
            raise Exception('Offset not found')

        # ME and secondary string are identified by the same index so we divide by 2
        index = int(index / 2)

        # Read 4 bytes (32bit int) that is the offset in a ROOT file
        offset = int.from_bytes(offsets_buf[index:index + 4], byteorder='little')
        # For now, hardcode, this functionality changed anyways
        offset = 57491238

        root_obj = self.__read_root_object(filename, '', offset)

        renderer = DQMRenderer()
        with await renderer.get_context() as context:
            png, error = context.render_histo(root_obj, [], name = path, spec='', width=width, height=height, efficiency = bool(False), streamerfile = filename.encode('utf-8'))
            return png
    
    def __read_root_object(self, filename, me_path, offset):
        """Returns binary representation of ROOT object wether it's a histogram or string ME."""

        filename = filename.replace('root://eoscms.cern.ch/', '')
        file = uproot.open(filename)
        cur = uproot.source.cursor.Cursor(offset)
        key = uproot.rootio.TKey.read(file.source, cur, file._context, None)
        cur = key._cursor.copied()
        buf = bytes(cur.bytes(key._source, key._fObjlen))

        if key._fClassName == b"TObjString":
            # In the case of a string, name contains the value in XML-like format.
            return key._fName
        else:
            # The format is <@length><kNewClassTag=0xFFFFFFFF><classname><nul><@length><2 bytes version><data ...
            # @length is 4byte length of the *entire* remaining object with bit 0x40 (kByteCountMask)
            # set in the first (most significant) byte. This prints as "@" in the dump...
            # the data inside the TKey seems to have the version already.
            classname = key._fClassName
            totlen = 4 + len(classname) + 1 + len(buf)
            head = struct.pack(">II", totlen | 0x40000000, 0xFFFFFFFF)
            return head + classname + b'\0' + buf