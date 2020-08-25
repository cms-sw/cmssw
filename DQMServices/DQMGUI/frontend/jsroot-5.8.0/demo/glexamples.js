examples_main = {
  TGeo: [
    {
      name: 'logo',
      asurl: true,
      file: 'geom/rootgeom.root',
      item: 'simple1;1',
      title: 'Geometry from rootgeom.C example',
    },
    {
      name: 'prim',
      file: 'geom/geodemo.root',
      layout: 'vert333',
      items: ['box', 'para', 'arb8', 'trap', 'gtra', 'trd1', 'trd2', 'xtru'],
      opts: ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z', 'z'],
      title: 'Basic TGeo primitives from tutorials/geom/geodemo.C',
    },
    {
      name: 'prim2',
      file: 'geom/geodemo.root',
      layout: 'vert333',
      items: [
        'cone',
        'coneseg',
        'tube',
        'tubeseg',
        'ctub',
        'eltu',
        'pcon',
        'pgon',
      ],
      opts: ['z', 'z', 'z', 'z', 'z', 'z', 'z', 'z'],
      title: 'Basic TGeo primitives from tutorials/geom/geodemo.C',
    },
    {
      name: 'prim3',
      file: 'geom/geodemo.root',
      layout: 'grid3x2',
      items: ['sphere', 'torus', 'parab', 'hype', 'comp'],
      opts: ['z', 'z', 'z', 'z', 'z'],
      title: 'Basic TGeo primitives from tutorials/geom/geodemo.C',
    },
    {
      name: 'comp',
      url:
        '?nobrowser&file=$$$geom/geodemo.root+&layout=grid2x2&items=[comp;1/Left,comp;1/Right,comp;1,comp;1]&opts=[az,az,comp_az,az]',
      title: 'Components of composite shape',
    },
    {
      name: 'building',
      asurl: true,
      file: 'geom/building.root',
      item: 'geom;1',
      opt: 'allz',
      title: 'Geometry from tutorials/geom/building.C',
    },
    {
      name: 'alice',
      asurl: true,
      file: 'https://root.cern/files/alice2.root',
      item: 'Geometry;1',
      opt: 'macro:https://root.cern/js/files/geomAlice.C;black',
      title: 'ALICE geometry',
    },
    {
      name: 'atlas',
      asurl: true,
      file: 'https://root.cern/files/atlas.root',
      item: 'atlas;1',
      opt: 'dflt;black',
      title: 'ATLAS geometry',
    },
    {
      name: 'cms',
      asurl: true,
      file: 'https://root.cern/files/cms.root',
      item: 'cms;1',
      opt: 'macro:https://root.cern/files/cms_cmse.C;clipxyz;black',
      title: 'CMS geomtery',
    },
    {
      name: 'lhcb',
      asurl: true,
      file: 'https://root.cern/files/lhcbfull.root',
      item: 'Geometry;1',
      opt: 'all;dflt;black',
      title: 'LHCb geometry',
    },
    {
      name: 'eve',
      asurl: true,
      json: 'geom/evegeoshape.json.gz',
      title: 'Example of drawing snapshot of volumes from EVE',
    },
    {
      name: 'tracks',
      url:
        '?nobrowser&json=$$$geom/evegeoshape.json.gz&file=$$$geom/eve_tracks.root&item=evegeoshape.json.gz+eve_tracks.root/tracks;1',
      title: 'Overlap of geometry with tracks, read from separate file',
    },
    {
      name: 'tracks+hits',
      url:
        '?nobrowser&json=$$$geom/simple_alice.json.gz&file=$$$geom/tracks_hits.root&item=simple_alice.json.gz+tracks_hits.root/tracks;1+tracks_hits.root/hits;1&opt=black',
      title:
        'Overlap of simple ALICE geometry with tracks and hits, read from separate file',
    },
    {
      name: 'proj',
      url:
        '?nobrowser&layout=h12_21&files=[https://root.cern/files/alice_ESDgeometry.root,$$$geom/eve_tracks.root]&items=[[[0]/Gentle,[1]/tracks],[0]/Gentle,[0]/Gentle]&opts=[main;black,projz;black,projx;black]',
      title: 'Simple ALICE geometry and two projections',
    },
    {
      name: 'overlap',
      asurl: true,
      file: 'geom/overlap.root',
      item: 'default/Overlaps/ov00010',
      itemfield: 'Overlaps/ov00010',
      title: 'example of TGeoOverlap',
    },
    {
      name: 'half',
      json: 'geom/comp_half.json.gz',
      title: 'Use of TGeoHalfSpace for building composites',
    },
    {
      name: 'atlas_cryo',
      asurl: true,
      file: 'https://root.cern/files/atlas.root',
      item: 'atlas;1',
      opt: 'macro:https://root.cern/files/atlas_c\
ryo.C',
    },
    {
      name: 'atlas_simple',
      asurl: true,
      json: 'geom/simple_atlas.json.gz',
      opt: 'ac',
    },
    { name: 'star', asurl: true, json: 'geom/star_svtt.json.gz' },
    { name: 'hades', asurl: true, json: 'geom/hades.json.gz' },
    { name: 'babar', asurl: true, json: 'geom/babar_emca.json.gz' },
    {
      name: 'alice_simple',
      asurl: true,
      json: 'geom/simple_alice.json.gz',
      title: 'simple alice geomtery',
    },
    {
      name: 'Dipole',
      url:
        '?nobrowser&file=https://root.cern/files/alice2.root&item=Geometry;1/ALIC/Dipole_1',
      title: 'Part of volumes from ge\
o manager',
    },
    {
      name: 'tank',
      asurl: true,
      file: 'https://root.cern/files/tank.root',
      item: 'geom;1',
      opt: 'z;rotate',
      title: 'Just for fun',
    },
    {
      name: 'lego',
      asurl: true,
      file: 'https://root.cern/files/lego.root',
      item: 'geom;1',
      opt: 'z;rotate',
      title: 'Just for fun',
    },
    {
      name: 'cheon',
      asurl: true,
      file: 'https://root.cern/files/cheongwadae.root',
      item: 'geom;1',
      opt: '',
      title: 'One more building',
    },
    {
      name: 'proj2',
      url:
        '?nobrowser&layout=h21_12&files=[https://root.cern/files/alice_ESDgeometry.root,$$$geom/eve_tracks.root]&items=[[0]/G\
entle,[0]/Gentle,[[0]/Gentle,[1]/tracks]]&opts=[projz,projx,main;black]',
      title: 'Place main drawing not on the first place',
    },
  ],
};
