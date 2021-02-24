from os.path import dirname, basename, isfile, join
import glob

from .offline_config import LAYOUTS as OFFLINE_LAYOUTS
from .online_config import LAYOUTS as ONLINE_LAYOUTS

directories_path = join(dirname(__file__) + '/layouts')
modules = glob.glob(join(directories_path, "*.py"))
__all__ = [join(basename(dirname(f)), basename(f)[:-3]) for f in modules if isfile(f) and not f.endswith('__init__.py')]
# Import all files in this folder
from .layouts import *
