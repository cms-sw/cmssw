import os
import time
from inspect import getframeinfo, stack

class Timed():
    """A helper that will measure wall clock time between enter and exit methods."""

    def __init__(self):
        caller = getframeinfo(stack()[1][0])
        self.filename = os.path.basename(caller.filename)
        self.lineno = caller.lineno

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        elapsed = elapsed * 1000
        print('%s:%s - %s ms' % (self.filename, self.lineno, elapsed))


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


def get_api_error(message):
    """Returns an object that is returned by the API to signify an error."""
    return { 'message': message }


def get_absolute_path(to=''):
    """
    Returns an absolute path to a specified directory/file. 
    Specified directory must be a relative path starting at DQMServices/DQMGUI/python/
    """

    base = os.environ.get('CMSSW_BASE')
    directory = 'src/DQMServices/DQMGUI/python/'
    if base:
        return os.path.join(base, directory, to)
    return os.path.join(directory, to)


def binary_search(array, target, key=None, decode=True):
    """
    Binary search implementation. Returns an index of an element if found, otherwise -1.
    if key is passed, value will be taken by that key from every item in the array.
    if decode is True, every item in the array is utf-8 decoded.
    """

    first = 0
    last = len(array) - 1

    while first <= last:
        mid = (first + last)//2

        if key:
            current = array[mid][key]
        else:
            current = array[mid]

        if decode:
            current = current.decode('utf-8')

        if current == target:
            return mid
        else:
            if target < current:
                last = mid - 1
            else:
                first = mid + 1
    return -1