##Version 2.0 - 2018/07/19
    - First version with C++11 enforcement
    - Support for property list
    - Support for Chunking
    - Support for Compression / Deflate
    - Fix: missing move constructor for properties
    - Fix: typo in MPI IO driver
    - Fix: several typo fixes
    - Fix: Add missing include

## Version 1.5 - 2018/01/06
    - SliceTraits::read split in two overloads, the first one for plain C arrays
      and the second one for other types.
    - Add support for complex number
    - Add exist() method to the API 
    - Will be last release before 2.0 and enforcement of C++11 

## Version 1.4 - 2017/08/25
	- Support id selection for the `select` function
	- Suport STL containers of const elements
	- Support scalar values and strings management
	- Fix attribute assignment issue #40
    - Fix Object assignment operator missing unref (possible memory leak )
    - Introduce SilenceHDF5 for HDF5 error report
    - Fix a unit test issue with SilenceHDF5


## Version 1.3 - 2017/06/21
    - Minor fixes

## Version 1.2 - 2017/04/03
	- Add Attribute support for Dataset
	- Extend testing of Attribute support
	- Fix issue related to multiple definitions in default driver
	- Add more examples about attribute support

## Version 1.1 - 2017/03/23
    - Add support and examples for Parallel HDF5
    - Initial implementation for H5 Properties
    - Support for Attributes 
    - Improve documentation
    - Add example for boost.Ublas matrix support


## Version 1.0 - Init
	- Initial release
