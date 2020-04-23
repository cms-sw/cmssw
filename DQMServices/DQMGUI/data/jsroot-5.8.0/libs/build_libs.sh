#!/bin/bash

# builds minified versions of used libs

minifier=/d/closure-compiler-v20190301.jar

# seems to be, dat.gui was minified more efficient than we can do
# java -jar $minifier --js dat.gui.js --js_output_file ../scripts/dat.gui.min.js

java -jar $minifier --js rawinflate.js --js_output_file ../scripts/rawinflate.min.js
