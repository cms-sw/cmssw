# DQMServices/DQMGUI

_Note: This is not the production DQMGUI._

This package contains experimental code for a DQMGUI similar to https://cmsweb.cern.ch/dqm/offline/ , but as part of CMSSW.

There are multiple relevant parts:
- The _render service_ in `bin/render.cc`, extracted from the classic DQMGUI: https://github.com/rovere/dqmgui
- The _render plugins_ in `plugins/`, traditionally hosted on https://github.com/dmwm/deployment/tree/master/dqmgui/style
- A web server (TBD).

## The render service

The histogram rendering using ROOT and the render plugins is done in a separate process in the classic DQMGUI. This package contians a simplified version of this process. `render.cc` compiles into a standalone program, `dqmRender`, that listens on a UNIX socket. A client (e.g. the GUI webserver) can request rendering a histogram there, by sending a request consisting of some metadata (some of which is a relict of the past and not actually used) and an arbitrary number of ROOT objects serialized into a `TBufferFile` buffer: The fisrt is the main object, the remaining ones are reference histograms that will be overlayed in different colors. (Some code for rendering _built-in references_ stored with the main object might remain, but this mode is no longer supported -- in line with CMSSW no longer supporting refrence hisotgrams in the `MonitorElement`.) The response is a PNG-compressed bitmap. All messages use a simple framing format of first sending the length, then the actual data, sometimes nested.

A client that implements this protocol is implemented in `python/render.py`.

The render process is single-threaded and does not do any IO apart from the UNIX socket. Many of them can be launched in parallel. They might crash now and then (because ROOT), so some  precautions should be taken to restart them if they fail.

### The render plugins

Render plugins are C++ classes that can modify how histograms are displayed. They are located in the `plugins/` folder, even though they are not EDM plugins. The render plugins are loaded dynamically by the render service (typically on startup, by passing the name of a `.so` with renderplugins to load). We have quite a lot of them, and they are the main reason to keep using this renderer (compared to e.g. switching to JSROOT).

### Compiling this code

The `scram` `BuildFile`s should do everything. But it is also not too hard to do it without scram, given a working installation of ROOT. There are no dependencies on other parts of CMSSW.

`render.cc` needs ROOT includes and a lot of ROOT libraries for linking, as well as `libpng`, which is a bit picky about versions (`-lpng15` was required here to work). It also needs to be linked dynamically to the render plugin base class so that the plugin registration works. This is done using a rule to link to this package, `DQMServices/DQMGUI`, in `BuildFile.xml`.

The render plugins need to register with their base class so that they can be called. This is done using a global variable in `DQMRenderPlugin.cc`. It is located in `src/` to get it compiled into a shared library, that can then share this state between the renderer and the render plugins.

The render plugins are compiled separately in `plugins/` and linked dynamically against `DQMRenderPlugin.cc` (via `DQMServices/DQMGUI` in the `BuildFile`). This results in a new shared library `.so`, which can then be dynamically loaded at runtime in `render.cc` (via `dlopen`), and all the plugins will automatically register. The render plugins are also linked against some other stuff from ROOT that they might need. They could actually depend on other CMSSW code now where it makes sense (e.g. detector geometries).

There is some hacky code in `render.py` that locates the `.so` with the render plugins and passes it to `render.cc` as a command line argument.

