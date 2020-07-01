"""
This file contains data classes/named tuples for most of the types
used throughout the DQM GUI application.
"""

from enum import IntEnum
from collections import namedtuple


Sample = namedtuple('Sample', ['run', 'dataset', 'lumi'])
SampleFull = namedtuple('SampleFull', ['dataset', 'run', 'lumi', 'file', 'fileformat'])
RootDir = namedtuple('RootDir', ['name', 'me_count'])
RootObj = namedtuple('RootObj', ['name', 'path', 'layout', 'qteststatuses'])
RootDirContent = namedtuple('RootDirContent', ['dirs', 'objs'])

# type can be one of those: Flag, Int, Float, XMLString, QTest or one of ROOT histogram types (TH*)
# Only type is non-optional parameter
MEInfo = namedtuple('MEInfo', ['type', 'seekkey', 'offset', 'size', 'value', 'qteststatus'], defaults=[0, 0, -1, None, 0])

EfficiencyFlag = namedtuple('EfficiencyFlag', ['name'])
ScalarValue = namedtuple('ScalarValue', ['name', 'type', 'value'])
QTest = namedtuple('QTest', ['name', 'qtestname', 'status', 'result', 'algorithm', 'message'])


class FileFormat(IntEnum):
    """An enumeration of all possible file formats that can be imported and used."""
    NONE = 0
    DQMCLASSIC = 1
    DQMIO = 2
    PROTOBUF = 3


class MEDescription:
    """Full description of a monitor element containing a run, dataset and full path to the ME."""

    def __init__(self, dataset, path, run, lumi=0):
        if dataset == None or path == None or run == None :
            raise Exception('dataset, path and run must be provided to MEDescription.')

        self.dataset = dataset
        self.path = path
        self.run = run
        self.lumi = lumi


class RenderingInfo:
    """Information needed to render a histogram"""

    def __init__(self, filename, fileformat, path, me_info, root_object=None):
        if filename == None or fileformat == None or path == None or me_info == None:
            raise Exception('filename, fileformat, path and me_info must be provided to RenderingInfo.')

        self.filename = filename
        self.fileformat = fileformat
        self.path = path
        self.me_info = me_info
        self.root_object = root_object


class RenderingOptions:
    """
    Contains options that will be passed down to the out of process renderer.
    If json is set to True, renderer will return JSON representation of the histogram.
    """

    def __init__(
        self, width=266, height=200, stats=True, normalize=True, error_bars=False,
        xtype=None, ytype=None, ztype=None,
        xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
        ref_labels=[], draw_opts=None, efficiency=False, json=False
    ):
        self.width = width
        self.height = height
        self.stats = stats
        self.normalize = normalize
        self.error_bars = error_bars
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.draw_opts = draw_opts
        self.efficiency = efficiency
        self.json = json

        self.xtype = None
        self.ytype = None
        self.ztype = None

        self.ref_labels = ref_labels

        if xtype == 'lin' or xtype == 'log':
            self.xtype = xtype
        if ytype == 'lin' or ytype == 'log':
            self.ytype = ytype
        if ztype == 'lin' or ztype == 'log':
            self.ztype = ztype


    @classmethod
    def from_dict(cls, dictionary):
        """Creates an instance of RenderingOptions from a dict containing URL variables."""

        return RenderingOptions(
            width = int(dictionary.get('w', 266)),
            height = int(dictionary.get('h', 200)),
            stats = str(dictionary.get('stats', 'true')) == 'true',
            normalize = str(dictionary.get('norm', 'true')) == 'true',
            error_bars = str(dictionary.get('errors', 'false')) == 'true',
            draw_opts = dictionary.get('drawopts', None),
            xtype = dictionary.get('xtype', None),
            ytype = dictionary.get('ytype', None),
            ztype = dictionary.get('ztype', None),
            xmin = cls.__get_float_or_none(dictionary, 'xmin'),
            xmax = cls.__get_float_or_none(dictionary, 'xmax'),
            ymin = cls.__get_float_or_none(dictionary, 'ymin'),
            ymax = cls.__get_float_or_none(dictionary, 'ymax'),
            zmin = cls.__get_float_or_none(dictionary, 'zmin'),
            zmax = cls.__get_float_or_none(dictionary, 'zmax'),
            ref_labels = dictionary.getall('reflabel', []),
        )

    
    @classmethod
    def from_dict_legacy(cls, dictionary):
        """Creates an instance of RenderingOptions from a dict containing URL variables."""

        return RenderingOptions(
            width = int(dictionary.get('w', 266)),
            height = int(dictionary.get('h', 200)),
            stats = int(dictionary.get('showstats', 1)) == 1,
            normalize = str(dictionary.get('norm', 'True')) == 'True',
            error_bars = int(dictionary.get('showerrbars', 0)) == 1,
            draw_opts = dictionary.get('drawopts', None),
            xtype = dictionary.get('xtype', None),
            ytype = dictionary.get('ytype', None),
            ztype = dictionary.get('ztype', None),
            xmin = cls.__get_float_or_none(dictionary, 'xmin'),
            xmax = cls.__get_float_or_none(dictionary, 'xmax'),
            ymin = cls.__get_float_or_none(dictionary, 'ymin'),
            ymax = cls.__get_float_or_none(dictionary, 'ymax'),
            zmin = cls.__get_float_or_none(dictionary, 'zmin'),
            zmax = cls.__get_float_or_none(dictionary, 'zmax'),
            ref_labels = dictionary.getall('reflabel', []),
        )


    @classmethod
    def __get_float_or_none(cls, dictionary, key):
        """
        Returns item from dictionare at key, formatted as a float. 
        If key doesn't exists or it can't be converted to float, returns None.
        """
        value = dictionary.get(key, None)
        if value:
            try:
                return float(value)
            except:
                pass
        return None
        

    def get_spec_string(self):
        """
        Formats options into a single string that renderer can understand.
        Sample: showstats=0;showerrbars=1;norm=False;xtype=lin;ytype=lin;ztype=log;xmin=1;xmax=2;ymin=3;ymax=4;zmin=5;zmax=6;drawopts=abc
        """

        spec = f'h={self.height};w={self.width};'
        if not self.stats:
            spec += 'showstats=0;'
        if not self.normalize:
            spec += 'norm=False;'
        if self.error_bars:
            spec += 'showerrbars=1;'
        if self.xtype:
            spec += f'xtype={self.xtype};'
        if self.ytype:
            spec += f'ytype={self.ytype};'
        if self.ztype:
            spec += f'ztype={self.ztype};'
        if self.xmin:
            spec += f'xmin={self.xmin};'
        if self.xmax:
            spec += f'xmax={self.xmax};'
        if self.ymin:
            spec += f'ymin={self.ymin};'
        if self.ymax:
            spec += f'ymax={self.ymax};'
        if self.zmin:
            spec += f'zmin={self.zmin};'
        if self.zmax:
            spec += f'zmax={self.zmax};'
        if self.json:
            spec += 'json=True;'
        if self.draw_opts:
            # TODO: sanitize draw_opts properly
            sanitized_draw_opts = self.draw_opts.replace(';', '')
            spec += f'drawopts={sanitized_draw_opts};'
        if self.ref_labels:
            # For now we support up to 9 references
            # TODO: We could set reflabel0 (for the base histogram) but that needs support from frontend.
            for i in range(min(9, len(self.ref_labels))):
                spec += f'reflabel{i+1}={self.ref_labels[i]};'

        if spec.endswith(';'):
            spec = spec[:-1]

        return spec
