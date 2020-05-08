
class MEDescription:
    """Full description of a monitor element containing a run, dataset and full path to the ME."""

    def __init__(self, run, dataset, path):
        if run == None or dataset == None or path == None:
            raise Exception('run, dataset and path must be provided to MEDescription.')

        self.run = run
        self.dataset = dataset
        self.path = path


class MERenderingInfo:
    """Information needed to render a histogram"""

    def __init__(self, filename, path, me_info, root_object=None):
        if filename == None or path == None or me_info == None:
            raise Exception('filename, path and me_info must be provided to MERenderingInfo.')

        self.filename = filename
        self.path = path
        self.me_info = me_info
        self.root_object = root_object


class PathUtil:
    """This helper class provides methods to handle common ME path related operations."""

    class PathSegment:
        """
        Represents one segment of a path. For example dir2 is a segment of this path: dir1/dir2/file
        If is_file is True, this segment represents a file, otherwise this segment represents a directory.
        """
        def __init__(self, name, is_file):
            self.name = name
            self.is_file = is_file


    def __init__(self, path=None):
        self.set_path(path)

    
    def set_path(self, path):
        self.path = path


    def subsequent_segment_of(self, subpath):
        """
        Returns a closest segment of path inside subpath.
        If path is a/b/c/d/file and subpath is /a/b/c function will return (d, False).
        If path is a/b/c/d/file and subpath is /a/b/c/d function will return (file, True).
        If subpath is not part of path, function will return None.
        """

        if self.path.startswith(subpath):
            names = self.path[len(subpath):].split('/')
            if len(names) == 1: # This is an ME
                return self.PathSegment(names[0], is_file=True)
            else: # This is a folder
                return self.PathSegment(names[0], is_file=False)
        else:
            return None


        