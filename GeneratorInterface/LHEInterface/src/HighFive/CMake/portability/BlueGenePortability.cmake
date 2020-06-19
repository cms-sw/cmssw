##
## Portability check on Blue Gene Q environment

include(CompilerFlagsHelpers)

if(IS_DIRECTORY "/bgsys" AND (NOT DEFINED BLUEGENE) )
    set(BLUEGENE TRUE)
endif()


if(BLUEGENE AND CMAKE_C_COMPILER_IS_XLC )
	# define library type to static on BGQ
	set(COMPILE_LIBRARY_TYPE "STATIC")
	## Blue Gene/Q do not support linking with MPI library when compiled with mpicc wrapper
        ## we disable any MPI_X_LIBRARY linking and rely on mpicc wrapper
	set(MPI_LIBRARIES "")
	set(MPI_C_LIBRARIES "")	
	set(MPI_CXX_LIBRARIES "")

	## static linking need to be forced on BlueGene
	# Boost need a bit of tuning parameters for static linking
	set(Boost_NO_BOOST_CMAKE TRUE)
    set(Boost_USE_STATIC_LIBS TRUE)	

	#enforce static linking for hdf5
	set(HDF5_USE_STATIC_LIBRARIES TRUE)

else()

if(NOT DEFINED COMPILE_LIBRARY_TYPE)
    set(COMPILE_LIBRARY_TYPE "SHARED")
endif()

endif()



