# Interactive Tracer Log Viewer

The viewer has a HTTP server component, implemented in Python using the `http.server` module, and a JavaScript client component.

## Usage

1. Copy the output of `edmTracerCompactLogViewer.py` and the files from `FWCore/Services/web` into a directory. The server lists all `.json` files in a drop-down list.
2. Run the server script (`server.py`) using Python.
3. Access the server in your web browser by navigating to `http://localhost:65432`.

### Specifying the Port

You can specify the port to serve on by using the `--port` argument:

```sh
python server.py --port 8080
```

## Viewer layout
The viewer is composed to two main areas, the top is the timing viewer and at the bottom shows information about a selected time block. The top time viewer is further divided into three parts. On the left is the macro scoping grouping of framework activity types into Global and Stream activities. If the _module centric_ option was chosen then this area is also divided by each module activity which is sorted based on most time used to least time used. At the bottom is the measurement of time since the start of the job. The main area shows the blocks of time spend doing various work within the Framework.

Within each macro grouping of activities (i.e Global or Stream) the work being done associated to that activity are broken into non overlapping rows. The top row is always related to the larger scale activity (e.g. the total time the framework spent processing a Begin Run or an Event). Below that row is the time individual modules spent within that larger scale activity. The association of a given time block of work to a given row within a group has no inherent meeting, the rows are just meant to avoid overlapping time blocks within the same larger scale activity.

Within the Global grouping, multiple concurrent running activities (e.g. LuminosityBlocks) are assigned to their own rows with their related module timings appearing in the rows directly below them.

## Using the viewer
Single clicking on a block within the main area will select that block and display information about the block into the Selection area at the bottom of the window. Clicking outside a block within the main area will deselect any selected item.

The time shown in the main view can be zoomed in/out by holding down the `control` key and then using a 'scroll wheel' (either a physical wheel on a mouse or by using a two finger drag on a track pad) within the display of the main view or the time measurement which is just below the main view. One can also do a drag left/right or a 'scroll wheel' left/right within the time measurment window to change the window of time being shown in the main view.
The area displayed in the main view can be changed by doing a drag or 'scroll wheel' operation within the main view.
The row name view on the left hand side of the display can be changed by doing a drag or 'scroll wheel' up/down operation within the view.
