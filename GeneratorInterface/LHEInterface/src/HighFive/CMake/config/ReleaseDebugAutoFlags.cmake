# ReleaseDebugAutoFlags.cmake
# 
# Release / Debug configuration helper
#
# License: BSD 3
#
# Copyright (c) 2016, Adrien Devresse
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 



## default configuration
if(NOT CMAKE_BUILD_TYPE AND (NOT CMAKE_CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
endif()


# Different configuration types:
#
# Debug : Optimized for debugging, include symbols
# Release : Release mode, no debuginfo 
# RelWithDebInfo : Distribution mode, basic optimizations for potable code with debuginfos
# Fast : Maximum level of optimization. Target native architecture, not portable code

include(CompilerFlagsHelpers)


set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_WARNING_ALL} ${CMAKE_C_OPT_NORMAL}")
set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_DEBUGINFO_FLAGS} ${CMAKE_C_WARNING_ALL} ${CMAKE_C_OPT_NONE} ${CMAKE_C_STACK_PROTECTION}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_DEBUGINFO_FLAGS} ${CMAKE_C_WARNING_ALL} ${CMAKE_C_OPT_NORMAL}")
set(CMAKE_C_FLAGS_FAST "${CMAKE_C_WARNING_ALL} ${CMAKE_C_OPT_FASTEST} ${CMAKE_C_LINK_TIME_OPT} ${CMAKE_C_GEN_NATIVE}")



set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_WARNING_ALL} ${CMAKE_CXX_OPT_NORMAL}")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_DEBUGINFO_FLAGS} ${CMAKE_CXX_WARNING_ALL} ${CMAKE_CXX_OPT_NONE} ${CMAKE_CXX_STACK_PROTECTION}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_DEBUGINFO_FLAGS} ${CMAKE_CXX_WARNING_ALL} ${CMAKE_CXX_OPT_NORMAL}")
set(CMAKE_CXX_FLAGS_FAST "${CMAKE_CXX_WARNING_ALL} ${CMAKE_CXX_OPT_FASTEST} ${CMAKE_CXX_LINK_TIME_OPT} ${CMAKE_CXX_GEN_NATIVE}")

