#!/bin/sh

dir=${1}

#cp index.php into all subdirectories
find ${dir} -mindepth 0 -type d -exec cp web/index.php {} \;
