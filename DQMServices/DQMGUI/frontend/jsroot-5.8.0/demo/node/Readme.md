# Use of JSROOT with Node.js

JSROOT is provided as npm module and always can be installed as:

    npm install jsroot

After this JSROOT functionality can be used from Node.js scripts via:

    var jsroot = require("jsroot");

Provided package.json file allows to use demos directly with local jsroot installation:

    npm install

Main motivation to use JSROOT from Node.js is creation of SVG files.
Example <makesvg.js> you will find in this directory. Just call it:

    node makesvg.js

JSROOT also provides possibility to read arbitrary TTree data without involving
any peace of native ROOT code. <tree.js> demonstrate such example:

    node tree.js
