var jsroot = require('jsroot');
var fs = require('fs');

console.log('JSROOT version', jsroot.version);

//Use embed into SVG images for drawing
//Required "npm install canvas" package
//
//jsroot.ImageSVG = true;

jsroot.OpenFile('https://root.cern/js/files/hsimple.root', function (file) {
  file.ReadObject('hpx;1', function (obj) {
    jsroot.MakeSVG(
      { object: obj, option: 'lego2,pal50', width: 1200, height: 800 },
      function (svg) {
        console.log('SVG size', svg.length);
        fs.writeFileSync('lego2.svg', svg);
      }
    );
  });
});
