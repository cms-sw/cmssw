var jsroot = require('jsroot');

jsroot.OpenFile('https://root.cern/js/files/hsimple.root', function (f) {
  f.ReadObject('ntuple;1', function (tree) {
    tree.Draw({ expr: 'px:py:pz', dump: true, numentries: 100 }, function (
      res
    ) {
      console.log('NumEntries', res.length);
      var sumx = 0,
        sumy = 0,
        sumz = 0;
      res.forEach(function (item) {
        sumx += item.x;
        sumy += item.y;
        sumz += item.z;
      });
      console.log('Mean x', sumx / res.length);
      console.log('Mean y', sumy / res.length);
      console.log('Mean z', sumz / res.length);
    });
  });
});
