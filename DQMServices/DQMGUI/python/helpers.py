import os
import time
from collections import namedtuple
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

    # Represents one segment of a path. For example dir2 is a segment of this path: dir1/dir2/file
    # If is_file is True, this segment represents a file, otherwise this segment represents a directory.
    PathSegment = namedtuple('PathSegment', ['name', 'is_file'])


    def __init__(self, path=None):
        self.set_path(path)

    
    def set_path(self, path):
        self.path = path


    def subsequent_segment_of(self, subpath):
        """
        Returns a closest segment of path inside subpath.
        If path is a/b/c/d/file and subpath is /a/b/c function will return (d, False).
        If path is a/b/c/d/file and subpath is /a/b/c/d function will return (file, True).
        is_file is True if subsequent segment is the last item in the path - a file (not a directory).
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


def get_base_release_dir():
    """
    Returns an absolute path to a CMSSW directory.
    If DQMServices/DQMGUI package is checked out, returns a path to user checkout.
    If DQMServices/DQMGUI package is not checked out, return a path to base CMSSW release.
    """

    user_path = os.environ.get('CMSSW_BASE')
    if user_path:
        if os.path.isdir(os.path.join(user_path, 'src/DQMServices/DQMGUI')):
            return user_path
    return os.environ.get('CMSSW_RELEASE_BASE')


def get_absolute_path(to=''):
    """
    Returns an absolute path to a specified directory or file. 
    Specified path must be a relative path starting at DQMServices/DQMGUI/python/
    """

    base = get_base_release_dir()
    directory = 'src/DQMServices/DQMGUI/python/'
    if base:
        return os.path.join(base, directory, to)
    return os.path.join(directory, to)


def binary_search(array, target, key=None):
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

        if current == target:
            return mid
        else:
            if target < current:
                last = mid - 1
            else:
                first = mid + 1
    return -1


def parse_run_lumi(runlumi):
    """
    Run/lumi pair is passed to the API in this format: run:lumi
    This method parses such string and returns a tuple (run, lumi).
    If lumi is not passed, it's assumed that it's value is 0.
    """
    
    if not runlumi:
        return None, None
    elif ':' in runlumi:
        parts = runlumi.split(':')
        if not parts[1]:
            return parts[0], 0
        else:
            return parts[0], parts[1]
    else:
        return runlumi, 0