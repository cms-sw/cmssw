# DQMServices/DQMGUI

_Note: This is not the production DQMGUI._

This package contains experimental code for a DQMGUI similar to https://cmsweb.cern.ch/dqm/offline/ , but as part of CMSSW.

There are multiple relevant parts:
- The _render service_ in `bin/render.cc`, extracted from the classic DQMGUI: https://github.com/rovere/dqmgui
- The _render plugins_ in `plugins/`, traditionally hosted on https://github.com/dmwm/deployment/tree/master/dqmgui/style
- A storage backend.
- A web server.
- A HTML frontend.

## The render service

The histogram rendering using ROOT and the render plugins is done in a separate process in the classic DQMGUI. This package contians a simplified version of this process. `render.cc` compiles into a standalone program, `dqmRender`, that listens on a UNIX socket. A client (e.g. the GUI webserver) can request rendering a histogram there, by sending a request consisting of some metadata (some of which is a relict of the past and not actually used) and an arbitrary number of ROOT objects serialized into a `TBufferFile` buffer: The fisrt is the main object, the remaining ones are reference histograms that will be overlayed in different colors. (Some code for rendering _built-in references_ stored with the main object might remain, but this mode is no longer supported -- in line with CMSSW no longer supporting refrence hisotgrams in the `MonitorElement`.) The response is a PNG-compressed bitmap. All messages use a simple framing format of first sending the length, then the actual data, sometimes nested.

A client that implements this protocol is implemented in `python/render.py`.

The render process is single-threaded and does not do any IO apart from the UNIX socket. Many of them can be launched in parallel. They might crash now and then (because ROOT), so some  precautions should be taken to restart them if they fail.

Since the `TBufferFile` data does not contain streamers, but we also don't want to repack all data into the latest format before rendering, the renderer has a mechanism to load streamers. This is done by passing a file name with the request, the renderer will simply open this file once (which reads the streamers as a side effect). This should only be done as needed, since it is quite slow (compared to the actual rendering).

### The render plugins

Render plugins are C++ classes that can modify how histograms are displayed. They are located in the `plugins/` folder, even though they are not EDM plugins. The render plugins are loaded dynamically by the render service (typically on startup, by passing the name of a `.so` with renderplugins to load). We have quite a lot of them, and they are the main reason to keep using this renderer (compared to e.g. switching to JSROOT).

### Compiling this code

The `scram` `BuildFile`s should do everything. But it is also not too hard to do it without scram, given a working installation of ROOT. There are no dependencies on other parts of CMSSW.

`render.cc` needs ROOT includes and a lot of ROOT libraries for linking, as well as `libpng`, which is a bit picky about versions (`-lpng15` was required here to work). It also needs to be linked dynamically to the render plugin base class so that the plugin registration works. This is done using a rule to link to this package, `DQMServices/DQMGUI`, in `BuildFile.xml`.

The render plugins need to register with their base class so that they can be called. This is done using a global variable in `DQMRenderPlugin.cc`. It is located in `src/` to get it compiled into a shared library, that can then share this state between the renderer and the render plugins.

The render plugins are compiled separately in `plugins/` and linked dynamically against `DQMRenderPlugin.cc` (via `DQMServices/DQMGUI` in the `BuildFile`). This results in a new shared library `.so`, which can then be dynamically loaded at runtime in `render.cc` (via `dlopen`), and all the plugins will automatically register. The render plugins are also linked against some other stuff from ROOT that they might need. They could actually depend on other CMSSW code now where it makes sense (e.g. detector geometries).

There is some hacky code in `render.py` that locates the `.so` with the render plugins and passes it to `render.cc` as a command line argument.

## The storage backend

The storage backend is based on legacy, `TDirectory` ROOT files. The code is in `rootstorage.py`. It keeps a SQLite database of metadata, about _samples_ (run/dataset/lumi, effectively _files_, for the legacy format), and _ME lists_, which represent the MEs present in a sample. These are stored compressed to make their size manageable. The ME list is built on first access; this makes it feasible to register all ~80000 files that we have on EOS at the moment as samples.

The storage backend is based on `uproot`, it never uses actual ROOT. To produce the `TBufferFile` format for the renderer, there is some custom byte-level packing code to add the required headers.

## The web server

There is a simple web server in `server.py`. It simply maps the classic DQMGUI API to the matching calls in the storage layer. It also does rendering using the render service.  The request parsing is very bad, so it fails in some cases and probably has lots of security issues.

The server is _threaded_, which allows it to do some of the IO waiting in parallel. But Python threading is not very efficient, so it limits at around 100 requests/s, which is less than the renderers could handle.

Please install python dependencies before running the code:
```
pip3 install -r DQMServices/DQMGUI/python/requirements.txt --user
```

Add python packages to a local directory. This is required for long time deployments because auth to AFS will eventually expire.
``` bash
python3 -m pip install -r requirements.txt -t .python_packages
```

The server is started like this:
```
dqmguibackend.sh
```

It will listen on `http://localhost:8889` (and you can't just change that, see below), and It will automatically create a DB file in `DQMServices/DQMGUI/data/` and populate it using data from EOS. 

## File formats

Currently there are three supported file formats:

* Legacy DQM TDirectory based ROOT files (1)
* DQMIO TTree based ROOT files (2)
* Protobuf based format used in Online live mode (3)

There will most probably be a format for online/live data streaming.

Format in the code is expressed as FileFormat enum.

## Adding new file format importer

`GUIImportManager` is responsible for importing blobs containing ME information into the database.
There are two types of blobs: `names_blob` and `infos_blob`. `names_blob` is `\n` separated, alphabetically ordered list of normalized full ME paths. All strings are represented as python3 binary strings. `infos_blob` contains a list MEInfo objects in exactly the same order as `infos_blob`. So in order to find out more information about a monitor element, we have to binary search for it in a sorted `names_blob` and access `MEInfo` from `infos_blob` at the same index. `get_rendered_image()` function in `GUIService` does this. Blobs are stored in the database compressed. `GUIBlobCompressor` service is responsible for compressing them for storage and uncompressing them for usage in the program.

In order to add new importer you have to do three things:

* Add a class into `python/importing/` folder following this naming convention: `<fileformat>_importer.py`.
  * This class has to have a single static coroutine `get_me_lists(cls, file, dataset, run, lumi):`
  * It returns a list which contains dicts. Keys of the dicts are (run, lumi) tuples and values are lists of tuples (me_path, me_info). Full structure: [(run, lumi):[(me_path, me_info)]]
* Add your new format to a `FileFormat` enum defined in `python/data_types.py`
* Modify `__pick_importer()` function in `GUIImportManager` to return an instance your importer when new file format is selected.


"""
        Returns a list which contains dicts. Keys of the dicts are (run, lumi) 
        tuples and values are lists of tuples (me_path, me_info). Full structure:
        [(run, lumi):[(me_path, me_info)]]
        me_path is normalized and represented as a binary string.
        We can return multiple (run, lumi) pairs because some file formats might 
        contain multiple runs/lumis in ine file.
        me_path, me_info will be saved as separete blobs in the DB.
        """

### Sample importer:

``` python
from data_types import MEInfo
class MyFormatImporter:
  @classmethod
  async def get_me_lists(cls, filename, dataset, run, lumi):
    # Actual reading of a file removed for brevity
    return {
      (run, lumi): [
        (b'/normalized/path/to/ME1', MEInfo(b'Float', value=float(1.23)), 
        (b'/normalized/path/to/ME2', MEInfo(b'TH1D', offset=123)
    ]}
```

## Adding new file format reader

After adding a new importer, a new reader has to be added as well. The process of adding a new reader is basically the same.

`GUIMEReader` is format agnostic service that will select a correct reader based on file format. The format specific service then opens up a ROOT files, reads an ME based on provided `MEInfo` and return one of these types: `ScalarValue`, `EfficiencyFlag`, `QTest`, `bytes`.

In order to add new reader you have to do three things:

* Add a class into `python/reading/` folder following this naming convention: `<fileformat>_reader.py`.
  * This class has to have a single static coroutine `read(cls, filename, me_info):`
  * It has to return one of the types listed above.
* Modify `__pick_reader()` function in `GUIMEReader` to return an instance your reader when new file format is selected.

### Sample reader:

``` python
from data_types import ScalarValue
class MyFormatReader:
    @classmethod
    async def read(cls, filename, me_info):
      # Actual reading of a file removed for brevity
      return ScalarValue(b'', b's', 'Value of the string ME')
```

## The HTML frontend

The frontend is developed here: https://github.com/cms-DQM/dqmgui_frontend

This package contains compiled code from there, which is served from the web server to get a working GUI. It is hardcoded to `localhost:8889`, so you can't easily change the port number in the server.

## API documentation

This is the future version of the DQM GUI API and it is preferred for all services over the legacy API.

**This API specification is not yet final and is subject to change!!!**

#### Samples endpoint

Returns run/dataset pairs available in the GUI. All arguments are optional. If lumi is not passed, 0 is assumed and only per run plots are returned. Passing -1, returns all per lumi samples and no per run samples.

`http://localhost:8889/api/v1/samples?run=295120&lumi=123&dataset=Run2017A`

```json
{
  "data": [
    {
      "run": 295120,
      "lumi": 123,
      "dataset": "/Cosmics/Run2017A-PromptReco-v1/DQMIO"
    },
    {
      "run": 295120,
      "lumi": 123,
      "dataset": "/StreamExpressCosmics/Run2017A-Express-v1/DQMIO"
    }
  ]
}
```

#### ROOT file directory listing endpoint

Run, full dataset and a path has to be provided in the URL.

If `layout` is `null`, ME is not coming from a layout. Otherwise, `layout` contains the name of the layout this ME comes from. 

`lumi` is optional. Passing 0 or not passing it at all returns per result.

`http://localhost:8889/api/v1/archive/316142/StreamExpress/Run2018A-Express-v1/DQMIO/PixelPhase1`
`http://localhost:8889/api/v1/archive/316142:123/StreamExpress/Run2018A-Express-v1/DQMIO/PixelPhase1`

```json
{
  "data": [
    {
      "subdir": "Summary"
    },
    {
      "subdir": "ClusterShape"
    },
    {
      "name": "num_feddigistrend_per_LumiBlock_per_FED",
      "path": "PixelPhase1/num_feddigistrend_per_LumiBlock_per_FED",
      "layout": null
    },
    {
      "name": "deadRocTotal",
      "path": "PixelPhase1/deadRocTotal",
      "layout": null
    },
  ]
}
```

#### Layouts endpoint

Returns all layouts with the same name. Used for quick collections.

`http://localhost:8889/api/v1/layouts?name=layout1`

```json
{
  "data": [
    {
      "source": "Hcal/TPTask/EtEmul/TTSubdet/HBHE",
      "destination": "Hcal/Layouts/EtEmul/TP/TTSubdet/HBHE_changed_name"
    }
  ]
}
```

#### Rendering endpoint

Renders a PNG of a histogram.

`http://localhost:8889/api/v1/render/316142:lumi/StreamExpress/Run2018A-Express-v1/DQMIO/PixelPhase1/EventInfo/reportSummaryMap?w=266&h=200&stats=false&norm=false&errors=true`

#### Overlay rendering endpoint

Overlays multiple (or one) histograms and renders an overlay to a PNG.

`http://localhost:8889/api/v1/render_overlay?obj=archive/316142/StreamExpress/Run2018A-Express-v1/DQMIO/PixelPhase1/EventInfo/reportSummary&obj=archive/316144/StreamExpress/Run2018A-Express-v1/DQMIO/PixelPhase1/EventInfo/reportSummary&w=266&h=200&stats=false&norm=false&errors=true`

#### New file registering endpoint

Registers new samples into the database.

`POST http://localhost:8889/api/v1/register`

HTTP request body:

`[{"dataset": "/a/b/c", "run": "123456", "lumi": "0", "file": "/a/b/c.root", "fileformat": 1}]`

`fileformat` is an integer. Please look at File formats section above for details.

### API endpoints for dealing with per lumisection data:

### Archive endpoint

Because not all plots are being saved per lumisection (depends on CMSSW configuration) and new per run plots are created in harvesting step, per run and per lumi directory listings of the same dataset will not match. For this reason, archive endpoint supports querying a directory listing of a specific lumisection:

`/api/v1/archive/run:lumi/dataset/me_path`

Lumi 0 indicates per run plots. If lumi is omitted and only run is provided, it's assumed that it's value is 0.

### Samples endpoint

`/api/v1/samples?run=317297&dataset=ZeroBias&lumi=555`

### Render endpoint

`/api/v1/render/run:lumi/dataset/me_path`

### Render overlay endpoint

`/api/v1/render_overlay?obj=archive/run:lumi/dataset/me_path`

### JSRoot render endpoint

`/api/v1/json/run:lumi/dataset/me_path`

### JSRoot overlay render endpoint

`/api/v1/json_overlay?obj=archive/run:lumi/dataset/me_path`

## List of lumisection available in dataset/run combination

`/api/v1/lumis/run/dataset`


## Getting DQMIO files

First you have to authenticate to access CMS data:

`voms-proxy-init --rfc --voms cms`

Getting a list of files and lumis:

`dasgoclient -query 'file run lumi dataset=/ZeroBias/Run2018C-12Nov2019_UL2018-v2/DQMIO'`

File needs to be on disk. In order to find out on which site the file resides:

`dasgoclient -query 'site file=/store/data/Run2018C/ZeroBias/DQMIO/12Nov2019_UL2018-v2/110000/E9FB467E-F8DF-4544-869F-F98E462FDF97.root'`

Copy desired file to local storage with a XRD redirector:

`xrdcp "root://cms-xrd-global.cern.ch//store/data/Run2018B/ZeroBias/DQMIO/12Nov2019_UL2018-v2/100000/0971E5EA-DA92-C249-96BD-1CE58A95C339.root" .`


# Random things

## Scalar types are stored as such:

INTs are saved as strings in this format: <objectName>i=value</objectName>
FLOATs are saved as strings in this format: <objectName>f=value</objectName>
STRINGs are saved as strings in this format: <objectName>s="value"</objectName>


## Desired UI functionality for per lumi data

We want to have two search fields (dataset and run) and a toggle switch. If toggle is off, we search only for per run data. If toggle is on, another text field appears (for lumi search). If that new text field is left empty, we search only for data that's available per lumi and return all lumis. If that text field is filled, we search for only for data that's available per lumi and filter lumis based on the contents of the field.

# Protocol buffers

``` bash
cd DQMServices/DQMGUI/python/
cp ../../Core/src/ROOTFilePB.proto protobuf/
protoc -I=protobuf --python_out=protobuf protobuf/ROOTFilePB.proto
```

# TODO

Backend related task list.

* ~~Live mode code~~
* ~~Provide async stream methods to IOService~~
* ~~Protobuf make parser async~~
* ~~Use bitwise operators for parsing variants in protobuf~~
* ~~Add QTests to protobuf output in CMSSW~~
* Live mode integration
* Cache invalidation in samples
* ~~Move efficiency flag to MEInfo~~
  * It's not worth it as it adds a couple of seconds to import time
* Clean up Scalar, EfficiencyFlag and QTest to OO hierarchy
* ~~Speed up linear search over sorted layouts~~
* Flavours?
  * The requirements and the need for this feature have to be reviewed
* Move common ME methods (like DQMCLASSICReader.parse_string_entry) to a separate location
* Check RelVal files are handled correctly
* ~~Make sure exceptions are logged to log file (atm the go to stderr)~~
* ~~Handle crashing import processes (prob. can't restart them, so at least crash the full server and wait for restart)~~
  * Whenever an import process crashes we restart ProcessPoolExecutor and return an error
* Check handling of XRD access failures (atm 500 response on the request, retry on next request -- might be good enough.)
* Make logging async
  * Will probably not increase perf by much, needs measuring
* ~~Renderer hangs when negative width/height is passed~~
* Validate samples in registration endpoint
* ~~Add timeout when interacting with the renderer~~
* ~~When renderer returns an error code, return an image with 500 status code.~~
* ~~Add QTest result support to the API~~
* Hanging/aborted requests don't get logged?
