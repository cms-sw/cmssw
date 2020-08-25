# Packaging of JSROOT code with gulp

This example shows how all JSROOT sources can be
merged togther with all dependend libraries.
This uses bower, Node.js and gulp.
Following steps are required:

## Correctly provide JSROOT location in bower.json file

In the example 5.3.0 version of JSROOT is used.

    "dependencies": { "jsroot": "^5.3.0" }

For experimental purposes one can configure local checkout of jsroot

    "dependencies": { "jsroot": "file:///home/user/git/jsroot/.git#dev" }

## Install packages with bower

    [shell] bower install

## Install gulp

    [shell] npm install

All used sub-packages listed in package.json file

## Create library with gulp

    [shell] node node_modules/gulp/bin/gulp.js

Source code of gulp script one can find in gulpfile.js
Script should produce "build/js/lib.js" and "build/css/lib.css"

## Open example web page

One can browser directly from the file system <file:///home/user/git/jsroot/demo/gulp/example.htm>

## Known issues

MathJax.js excluded from common library.
It will be loaded only when required from default location.
