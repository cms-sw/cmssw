"""

Unit tests for python conddblib framework.

"""

import unittest
import sys

from payload_tests import *
from shell_tests import *
from data_formats_tests import *
from querying_tests import *
from data_sources_tests import *

if __name__ == "__main__":
	unittest.main(verbosity=2)