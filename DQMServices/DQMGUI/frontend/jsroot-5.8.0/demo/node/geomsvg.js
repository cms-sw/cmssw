var jsroot = require('jsroot');
var fs = require('fs');

console.log('JSROOT version', jsroot.version);

// Use embed into SVG images for drawing
// Required "npm install canvas" package
//
// jsroot.ImageSVG = true;

jsroot
  .NewHttpRequest(
    'https://root.cern/js/files/geom/simple_alice.json.gz',
    'object',
    function (obj) {
      jsroot.MakeSVG({ object: obj, width: 1200, height: 800 }, function (svg) {
        console.log('SVG size', svg.length);
        fs.writeFileSync('alice_geom.svg', svg);
      });
    }
  )
  .send();
