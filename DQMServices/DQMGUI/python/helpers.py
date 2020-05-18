
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

