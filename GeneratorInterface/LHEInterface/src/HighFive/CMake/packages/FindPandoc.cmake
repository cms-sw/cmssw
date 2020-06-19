include(FindPackageHandleStandardArgs)

if(DEFINED PANDOC_EXECUTABLE)
  set(Pandoc_FIND_QUIETLY TRUE)
endif()

find_program(PANDOC_EXECUTABLE
  NAMES pandoc
  DOC "Pandoc - a universal document converter")

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Pandoc REQUIRED_VARS PANDOC_EXECUTABLE)

mark_as_advanced(PANDOC_EXECUTABLE)
