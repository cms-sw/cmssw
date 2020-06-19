# CompilerFlagsHelpers.cmake
# 
# set of Convenience functions for portable compiler flags
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



set(SUPPORTED_COMPILER_LANGUAGE_LIST "C;CXX")

## detect compiler
foreach(COMPILER_LANGUAGE ${SUPPORTED_COMPILER_LANGUAGE_LIST})

	if(CMAKE_${COMPILER_LANGUAGE}_COMPILER_ID STREQUAL "XL")
	  set(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_XLC ON)
	elseif(CMAKE_${COMPILER_LANGUAGE}_COMPILER_ID STREQUAL "Intel")
	  set(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_ICC ON)
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	  set(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_MSVC ON)
	elseif(${CMAKE_${COMPILER_LANGUAGE}_COMPILER_ID} STREQUAL "Clang")
	  set(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_CLANG ON)
	else(CMAKE_${COMPILER_LANGUAGE}_COMPILER_ID STREQUAL "GNU")
	  set(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_GCC ON)
	endif()

endforeach()




foreach(COMPILER_LANGUAGE ${SUPPORTED_COMPILER_LANGUAGE_LIST})

	# XLC compiler
	if(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_XLC) 

		# XLC -qinfo=all is awfully verbose on any platforms that use the GNU STL
		# Enable by default only the relevant one
		set(CMAKE_${COMPILER_LANGUAGE}_WARNING_ALL "-qformat=all -qinfo=lan:trx:ret:zea:cmp:ret")

		set(CMAKE_${COMPILER_LANGUAGE}_DEBUGINFO_FLAGS "-g")

		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NONE "-O0")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NORMAL "-O2")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_AGGRESSIVE "-O3")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_FASTEST "-O5")

		set(CMAKE_${COMPILER_LANGUAGE}_STACK_PROTECTION "-qstackprotect")

		set(CMAKE_${COMPILER_LANGUAGE}_POSITION_INDEPENDANT "-qpic=small")

		set(CMAKE_${COMPILER_LANGUAGE}_VECTORIZE "-qhot")

	# Microsoft compiler
	elseif(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_MSVC) 

		set(CMAKE_${COMPILER_LANGUAGE}_DEBUGINFO_FLAGS "-Zi")

		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NONE "")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NORMAL "-O2")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_AGGRESSIVE "-O2")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_FASTEST "-O2")

		set(CMAKE_${COMPILER_LANGUAGE}_STACK_PROTECTION "-GS")

		# enable by default on MSVC
		set(CMAKE_${COMPILER_LANGUAGE}_POSITION_INDEPENDANT "")


	## GCC, CLANG, rest of the world
	else() 

		set(CMAKE_${COMPILER_LANGUAGE}_WARNING_ALL "-Wall -Wextra")

		set(CMAKE_${COMPILER_LANGUAGE}_DEBUGINFO_FLAGS "-g")

		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NONE "-O0")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_NORMAL "-O2")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_AGGRESSIVE "-O3")
		set(CMAKE_${COMPILER_LANGUAGE}_OPT_FASTEST "-Ofast -march=native")

		set(CMAKE_${COMPILER_LANGUAGE}_STACK_PROTECTION "-fstack-protector")

		set(CMAKE_${COMPILER_LANGUAGE}_POSITION_INDEPENDANT "-fPIC")

		set(CMAKE_${COMPILER_LANGUAGE}_VECTORIZE "-ftree-vectorize")

		if(CMAKE_${COMPILER_LANGUAGE}_COMPILER_IS_GCC AND ( CMAKE_${COMPILER_LANGUAGE}_COMPILER_VERSION VERSION_GREATER "4.7.0") )
			set(CMAKE_${COMPILER_LANGUAGE}_LINK_TIME_OPT "-flto")
		endif()

		if( (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^ppc" ) OR ( CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^power" ) )
			## ppc arch do not support -march= syntax 
			set(CMAKE_${COMPILER_LANGUAGE}_GEN_NATIVE "-mcpu=native")
		else()
			set(CMAKE_${COMPILER_LANGUAGE}_GEN_NATIVE "-march=native")
		endif()
	endif()



endforeach()



