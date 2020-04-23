// this small macro is required to correctly read data from
//     https://root.cern/files/markus.root
// Seems to be, pool::Token is class from Gaudi Framework
// There is no streamer information stored in the file
// Some idea that is stored there can be found
//   http://svn.cern.ch/guest/gaudi/Gaudi/trunk/RootCnv/src/RootIOHandler.cpp, line 175
// There are 4 bytes mismatch - caused by problem with version reading in JSROOT
// Normally version for foreign classes stored with the checksum
// JSROOT searches for checksum and rall-back when streamer info not found
// This is a case for pool::Token class, therefore checksum should be skipped here

JSROOT.addUserStreamer('pool::Token', function (buf, obj) {
  obj._typename = 'pool::Token';
  buf.ntou4(); // skip version checksum
  obj.m_oid = {
    _typename: 'pair<int,int>',
    first: buf.ntoi4(),
    second: buf.ntoi4(),
  };
});
