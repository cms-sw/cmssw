
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

    def __init__(self, filename, path, meinfo):
        if filename == None or path == None or meinfo == None:
            raise Exception('filename, path and meinfo must be provided to MERenderingInfo.')

        self.filename = filename
        self.path = path
        self.meinfo = meinfo
        self.root_object = None

