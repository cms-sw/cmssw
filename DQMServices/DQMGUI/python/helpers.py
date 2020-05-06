
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

    def __init__(self, filename, path, me_info):
        if filename == None or path == None or me_info == None:
            raise Exception('filename, path and me_info must be provided to MERenderingInfo.')

        self.filename = filename
        self.path = path
        self.me_info = me_info
        self.root_object = None


class PathUtil:
    """This helper class provides methods to handle common ME paths related operations."""

    def __init__(self, path):
        if not path:
            raise Exception('path has to be provided')

        self.path = path

        