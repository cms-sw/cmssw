// function to extract tracks from ALICE_ESD tree
// tracks are in https://root.cern/files/alice_ESDs.root
// geometry in https://root.cern/files/alice_ESDgeometry.root

function extract_geo_tracks(tree, opt, call_back) {
  // as first argument, tree should be provided

  console.log('CALL extract_geo_tracks');

  var selector = new JSROOT.TSelector();

  selector.AddBranch('ESDfriend.fTracks.fPoints', 'pnts');

  var lst = JSROOT.Create('TList'),
    numentry = 0,
    numtracks = 0;

  selector.Process = function () {
    // function called for every entry

    var pnts = this.tgtobj.pnts;

    numentry++;

    // now converts AliTrackPointArray into TGeoTrack
    for (var p = 0; p < pnts.length; ++p) {
      numtracks++;
      var arr = pnts[p];
      if (!arr.fNPoints) continue;
      var track = JSROOT.Create('TGeoTrack');
      track.fNpoints = arr.fNPoints;
      track.fPoints = new Float32Array(track.fNpoints * 3);
      for (var k = 0; k < track.fNpoints; ++k) {
        track.fPoints[k * 3] = arr.fX[k];
        track.fPoints[k * 3 + 1] = arr.fY[k];
        track.fPoints[k * 3 + 2] = arr.fZ[k];
      }
      track.fLineWidth = 2;
      track.fLineColor = 3;
      lst.Add(track);
      if (numtracks > 100) return this.Abort(); // do not accumulate too many tracks
    }
  };

  selector.Terminate = function (res) {
    // function called when processing finishes
    console.log('Read done num entries', numentry, 'tracks', numtracks);
    JSROOT.CallBack(call_back, lst);
  };

  tree.Process(selector);
}

console.log('LOAD alice_esd.js JSROOT', JSROOT.version);
