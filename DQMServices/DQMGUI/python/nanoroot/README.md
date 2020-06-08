nanoroot
========

If `uproot` is Mirco-Python ROOT, this `nanoroot`.

The code in this package implement minimal read support for the ROOT container 
formats: `TFile`, `TDirectory`, `TTree`, and `TBufferFile`. It *does not* decode
serialized objects; only strings and numeric types. Objects can be passed around
as buffers, to be read using "real" ROOT. This largely avoids a dependence on
the ROOT "streamer" mechanism.

The motiviations to use this, rather than `uproot` are:

- `asyncio` support.
- Better performance when traversing large numbers of objects.
- More direct access to internal fields, to allow directly reading object data from ROOT files.
- Fewer dependencies: only optionally `pyxrootd` for XrootD IO and `async_lru` for the `TTree` basket cache.

Many things in this code use terminolgy and variable/type names straight from 
the ROOT source code. Typically these are the fields starting with `f*` and the
classes starting with `T*`.

This code uses `async` code in unusual spots; namely, there are many classes with
`async` `__getitem__`, some of which return generators to amke it worse. `await`
may be needed in surprising places.

To work around the restriction of not having `async` constructors, most classes
have an empty constructor and an `async` `load` method that replaces the
constructor. This method returns self, so it can be used like
 `key = await TKey().load(buf, addr)`.

tfile
-----

This is the most fundamental code for reading ROOT files.

A ROOT file consists of objects stored in `TKey` containers, which form a tree
structure described by "parent" references in the `TKey`s. The `TDirectory` 
objects contain references to the keys that they contain, but there is no 
support to actually read these. Instead, just list the full file content.

io
-----

All the other code expects to do IO via objects with and `async` `[...:...]` 
operator. This code implements two such interfaces: one to on-disk files using
`mmap`, and one using XrootD.

The main work here is a bridge beween the thread & callback based interface of
`pyxrootd` and Python3 `asyncio`.

tbufferfile
-----------

Reading objects from ROOT files would be pretty useless if one cannot actually
use them. This class implements *writing* the `TBufferFile::ReadObjectAny` framing format,
so that objects read from ROOT files using `nanoroot` can be passed to other 
ROOT implementations that can then decode these objects. Since the bare `TBufferFile`
format does not contain streamers, the code that attempts to unpack these buffers
either needs to already have the streamers (unlikely, since we might have read
older oder newer versions from the file) or needs to be given access to the
streamers of the original file.

ttree
-----

`TTree` is the most advanced ROOT container format. `TTree`s are essentially a
column-oriented database format: A table (`TTree`) consisting of rows (_entries_)
and columns (`TBranch`), which are stored independently of each other. Each
column is stored into compressed blocks (`TBasket`) of a variable number of entries
which are then stored in `TKey`s in the file.

Decoding this structure is a lot more complicated and fragile compared to the 
other code, since it is not possible without getting some information out of
serialized objects. This is done using pattern matching, which seems to work
most of the time. 

Unlike ROOT, this code needs an explicit schema description (since it can't read
the type data stored in the `TLeaf` objects). Therefore, the `TType` namespace
used for this does not exist in ROOT.

To allow for partial reading of files, there is also minimal code for decoding
`TDirectory`s; this is only suitable to load `TTree`s and in the special class
`TTreeFile`, which also has no equivalent in ROOT.

Once created from a file using `TTreeFile`, trees and branches can be accessed
using the `async []` operator, which will request remote data as needed.
