Common L1T utilities for demonstrators
======================================

This package contains a prototype implementation of subsystem-agnostic utility functions and
classes for converting between emulator input/output objects (i.e. C++ objects representing
tracks, calo clusters, muon TPs, vertices etc) and the I/O buffer text files (i.e. the files
that are used to store data that is loaded into or captured from FPGAs, and used to play/capture
data in HDL simulations). The motivation for these tools and their scope was briefly summarised
in the L1T meeting on 16th February (see slides [here](https://indico.cern.ch/event/1008519/contributions/4234188/attachments/2191061/3703176/l1tPhase2_cmsswBufferIO_20210216.pdf))

A brief tour of the code:

 * One example EDAnalyzer - `GTTFileWriter` - that creates text files for loading into
   input buffers of the GTT board (as well as reference files for outputs of GTT boards)
   * Source: `plugins/GTTFileWriter.cc`
   * Test cmsRun config: `test/gtt/createFirmwareInputFiles_cfg.py`
 * One example EDProducer - `GTTFileReader` - that reads text files that would be 
   produced by GTT vertex-finding FW
   * Source: `plugins/GTTFileReader.cc`
   * Test cmsRun config: `test/gtt/verifyFirmwareOutput_cfg.py`
 * Main utility classes:
    - `BoardData`: Represents the data stored in I/O buffers
    - `ChannelSpec`: Simple struct containing parameters that define a link's packet
      structure (e.g. TMUX period, gap between packets)
    - `EventData`: Represents the data corresponding to a single event, with links
      labelled using logical channel IDs (`LinkId` class instances) that are local
      to any given time slice (e.g. ["tracks", 0] -> ["tracks", 17] for links from
      TF). Used to provide an event-index-independent interface to the
      `BoardDataWriter` & `BoardDataReader` classes - i.e. to avoid any need to keep
      track of  `eventIndex % tmux` when using the reader & writer classes for boards
      whose TMUX period is less than any of their upstream systems.
    - `BoardDataReader`: This class ...
        1. reads a set of buffer files
        2. verifies data conforms to expected structure
        3. splits out each event; and
        4. returns data for each event in turn via `getNextEvent()` method
    - `BoardDataWriter`: Essentially, does the opposite of the reader -
        1. accepts per-event data via `addEvent` method
        2. concatenates events according to specified structure; and
        3. automatically writes out pending data to file whenever the limit on the
           number of frames per file is reached.

(Note: For simplicity this code has been put in its own package during development,
but this might not be its final location.)

Given the above, the contents of `GTTFileWriter.cc` and `GTTFileReader.cc`
should be mostly self-explanatory, but a couple of additional notes:

 * These `.cc` files each contain some hardcoded constants defining the link TM periods,
   TM indices, packet structure etc. that are used to create the `BoardDataReader`/`BoardDataWriter`
   instance (so that it can correctly separate/concatenate events).
   * At least some (if not all) of these types of constants should eventually be read from
     config files, to avoid e.g. needing to recompile code when the latency of an algo changes
 * The EDAnalyzer/EDProducer analyze/produce methods use functions from the 'codecs' directory
   to convert between EDM collections and vectors of `ap_uint<64>`.

Finally: Suggestions (and implementations) for improvements most welcome, but please get in
touch with Tom Williams about these before writing large amounts of code, to avoid duplicating
efforts or divergent solutions to the same problem.
