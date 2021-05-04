# Info about EMTF Unpacker

Original author: Andrew Brinkerhoff &lt;andrew.wilson.brinkerhoff@cern.ch&gt;

Last edit: September 11, 2017


## Documentation

* AMC13 DAQ Facilities (`AMC13_DAQ`)
  - docs/UpdatedDAQPath_2015-09-30.pdf
  - Contains the description of AMC13 and MTF7 header and trailer words
* CMS Muon Endcap Track Finder DAQ readout format (`EMTF_DAQ`)
  - docs/EMU_TF_DAQ_format_2017_03_17.docx
  - Contains the description of the EMTF MTF7 payload
* EMTF track address format for uGMT output
  - docs/emtf-gmt-output.docx
  - Contains description of EMTF information transmitted to uGMT
  - Encoded in RegionalMuonCand class in software
* Interface between track finders and uGMT for the upgraded 2016 trigger
  - docs/DN2015_017_v3.pdf
  - Contains description of pT, eta, phi, etc. scales and conventions between EMTF and uGMT

##  General description of algorithm

* Run inside L1Trigger/L1TMuonEndCap using the command: `cmsRun RunTrackFinder_data.py`
* Unpacking code is all located inside EventFilter/L1TRawToDigi
* EMTF data is unpacked into classes defined in DataFormats/L1TMuon
* EMTF-specific modules in EventFilter/L1TRawToDigi/plugins/implementations_stage2
* Order of EMTF unpacking:

```
   #  File                     <--->  Output class   <--->   Documentation
   ----------------------------------------------
   0  EMTFBlockHeaders.cc      <--->  AMC13Header    <--->   AMC13_DAQ page 4/7, lines  1 -  2, 2 64-bit words
   1  EMTFBlockHeaders.cc      <--->  AMC13Header    <--->   AMC13_DAQ page 4/7, lines  3 -  6, 1 64-bit word per input MTF7
      *** Loop over each input MTF7: ***
   2     EMTFBlockHeaders.cc   <--->  MTF7Header     <--->   AMC13_DAQ page 3/7, lines  1 -  2, 2 64-bit words
   3     EMTFBlockHeaders.cc   <--->  EventHeader    <--->   EMTF_DAQ  page 2/9, lines  1 - 12, 3 64-bit words (if MTF7 has tracks)
   4     EMTFBlockCounters.cc  <--->  Counters       <--->   EMTF_DAQ  page 3/9, lines  1 -  4, 1 64-bit word  (if MTF7 has tracks)
   5        EMTFBlockME.cc     <--->  ME, EMTFHit    <--->   EMTF_DAQ  page 4/9, lines  1 -  4, 1 64-bit words per CSC track
   6        EMTFBlockRPC.cc    <--->  RPC, EMTFHit   <--->   EMTF_DAQ  page 6/9, lines  1 -  4, 1 64-bit words per RPC track
   7        EMTFBlockSP.cc     <--->  SP, EMTFTrack  <--->   EMTF_DAQ  page 6/9, lines  1 -  8, 2 64-bit words per output track
                                      RegionalMuonCand
   8     EMTFBlockTrailers.cc  <--->  EventTrailer   <--->   EMTF_DAQ  page 8/9, lines  1 -  8, 2 64-bit words (if MTF7 has tracks)
   9     EMTFBlockTrailers.cc  <--->  MTF7Trailer    <--->   AMC13_DAQ page 3/7, line        4, 1 64-bit word
  10  EMTFBlockTrailers.cc     <--->  AMC13Header    <--->   AMC13_DAQ page 4/7, line       11, 1 64-bit word
  11  EMTFBlockTrailers.cc     <--->  AMC13Header    <--->   AMC13_DAQ page 4/7, line       12, 1 64-bit word
```

##  Bitwise operators
* "Word >> X" returns "Word" shifted right by X binary bits (X/4 hex bits). Thus when you read a bit out of "Word", 
  instead of reading bit Y you will read bit Y-X.
* "Word << X" returns "Word" shifted left by X binary bits (X/4 hex bits). This is basically multiplication by 2^X.
  Thus, if I say Y = 0x00f << 8, it is equivalent to Y = 0xf00.
* "& 0xff" reads the right-most 8 binary bits (2 hex bits), "& 0xf" the right-most 4, "& 0x7" the right-most 3, 
  "& 0x3" the right-most 2, and "& 0x1" the right-most 1 (i.e. bit 0)
* Similarly, "& 0x2" reads bit 1, "& 0x4" bit 2, "0x8" bit 3, "0x10" bit 4, "0x20" bit 5, "0x40" bit 6, "0x80" bit 7,
  "0x100" bit 8, "0x200" bit 9, "0x400" bit 10, "0x800" bit 11, "0x1000" bit 12, "0x2000" bit 13, "0x4000" bit 14, and "0x8000" bit 15
* "|=" is the bitwise "OR" assignment. As used here (to concatenate words), it is basically addition, since the 
  things being "OR-ed" do not have contents in the same bits.
