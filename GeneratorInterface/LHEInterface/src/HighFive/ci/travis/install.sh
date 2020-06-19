#!/bin/bash


# install all dependencies for linux targets
if [[ "${TRAVIS_OS_NAME}" == "linux" ]]; then 
	
	sudo apt update -y
	sudo apt install -y libhdf5-serial-dev libboost-all-dev cmake g++ ninja-build

fi

# install all dependencies for osx targets
if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then 
	
	brew install boost hdf5 cmake ninja
fi


