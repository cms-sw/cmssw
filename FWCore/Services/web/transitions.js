import {data} from './data.js';
//import data from './data.json' assert { type: 'json' };

const kBoxHeight = 15;
const kRowHeight = 20;

const kTypeOffset = 16
const kSourceColor = "#DD00DD";
const typeToColor = [
    "#000000", //destructor
    0, // unused -15
    0, //unused -14
    0, //unused -13
    "#FF0000", //endjob
    "#FF7777", //endstream
    "#BBBBBB", //write process block
    "#FFFFFF", //end process block
    0, //unused -8
    "#0000BB", //write end run
    "#0000FF", //global end run
    "#0000FF", //stream end run
    "#00BB00", //write end lumi
    "#00FF00", //global end lumi
    "#00FF00", //stream end lumi
    "#CCCC00", //clear event
    "#FFFF00", //event
    0, //unused 1
    "#007700", //stream begin lumi
    "#007700", //global begin lumi
    0, //unused 4
    "#000077", //stream begin run
    "#000077", //global begin run
    0, //unused 7
    "#BBBBBB", //access input process block
    "#FFFFFF", //begin process block
    kSourceColor, //open file
    "#FF7777", //begin stream
    "#FF0000", //begin job
    "#00FFFF", //sync
    "#007777", //sync enqueue
    kSourceColor, // getNextTransition
    "#DD00DD", //construction
    "#FF0000", //startup
]

const overlappingTransitions =[15]

const typeToName = [
    "destructor",
    0, //unused -15
    0, //unused -14
    0, //unused -13
    "end job",
    "end stream",
    "write process block",
    "end process block",
    0, //unused -8
    "write end run",
    "global end run", //global end run
    "stream end run", //stream end run
    "write end lumi", //write end lumi
    "global end lumi", //global end lumi
    "stream end lumi", //stream end lumi
    "clear event",
    "event", //event
    0, //unused 1
    "stream begin lumi", //stream begin lumi
    "global begin lumi", //global begin lumi
    0, //unused 4
    "stream begin run", //stream begin run
    "global begin run", //global begin run
    0, //unused 7
    "access input process block", //access input process block
    "begin process block", //begin process block
    "open file",
    "begin stream", //begin stream
    "begin job", //begin job
    "EventSetup synch", //sync
    "EventSetup sych enqueue", //sync enqueue
    "find next transition",
    "construction", //construction
    "startup", //startup
]

const activityToName = [  "prefetch",
                          "acquire",
                          "process",
                          "delayedGet",
                        "externalWork"]

const activityToColor = ["#FF5F1F", "#CC7722", null, "#FF4433", "#8B4513"];

window.onload = () => {
    
    //const response = await fetch("./data.json");
    //const data = await response.json();
    const left = document.querySelector('.name_div');
    const div = document.querySelector('.graph_div');
    const bottom = document.querySelector('.time_div');
    const graph = document.getElementById('graph_view');
    const graph_context = graph.getContext('2d');
    const name_view = document.getElementById('name_view');
    const name_context = name_view.getContext('2d');
    const time_view = document.getElementById('time_view');
    const time_context = time_view.getContext('2d');
    const selected_view = document.getElementById('selected_paragraph');
    const zoom_in_button = document.getElementById('zoom_in');
    const zoom_out_button = document.getElementById('zoom_out');
    let selected_item = null;
    selected_view.innerHTML = "Selected: [click on box in graph]";
    let mouse_is_down = false;
    
    let rows = 0;
    for( let grouping of data.transitions ) {
        for( let slot in grouping.slots ) {
            ++rows;
        }
    }

    const max_graph_height = kRowHeight*rows;
    
    //same scale as window coordinates
    let graph_vertical_offset = 0.;
    let minVisibleTime = 0.;
    let timeZoomFactor = 1.0;
    
    function maxTime() {
        let maxTime = 0;
        for( let grouping of data.transitions) {
            for( let slot of grouping.slots) {
                for (let transition of slot) {
                    if (maxTime < transition.finish) {
                        maxTime = transition.finish;
                    }
                }
            }
        }
        
        maxTime = Math.ceil(maxTime);
        
        const digits = maxTime.toString().length;
        const digitsToZeroOut = digits -2;
        if (digitsToZeroOut > 0) {
            for(let i = 0; i < digitsToZeroOut; ++i) {
                maxTime /= 10;
            }
            maxTime *=10;
            maxTime +=9;
            for( let i = 1; i < digitsToZeroOut; ++i) {
                maxTime *=10;
            }
            maxTime = Math.ceil(maxTime);
        }
        return maxTime;
    }
    
    const kEndTime = maxTime();
    
    function drawNames() {
        name_context.setTransform(1,0,0,1,0,0);
        name_context.fillStyle = "#AACD6E"
        name_context.fillRect(0,0,name_view.width,name_view.height);
        name_context.scale(1,1)
        name_context.fillStyle = "black"
        name_context.strokeStyle = "black"
        name_context.font = '9pt monospace';		//context.font = "50px";
        
        let offset = kRowHeight/2 + graph_vertical_offset;
        for( let grouping of data.transitions ) {
            name_context.fillText(grouping.name, 0, offset);
            offset += grouping.slots.length*kRowHeight;
        }
    }
    
    function timeStepPower() {
        let timeDecade = 0;
        let nSteps = time_view.width/kInitPixelsPerSecond/timeZoomFactor;
        while (nSteps < 3) {
            --timeDecade;
            nSteps *=10.;
        }
        while (nSteps > 20) {
            ++timeDecade;
            nSteps /=10.;
        }
        return timeDecade;
    }
    
    function drawTime() {
        time_context.save()
        time_context.setTransform(1,0,0,1,0,0);
        time_context.fillStyle = "#CD6E6E";
        time_context.fillRect(0,0,time_view.width,time_view.height);
        time_context.restore();
        time_context.scale(1,1);
        const end = kEndTime;
        const timePower = timeStepPower();
        const stepSize = Math.pow(10., timePower);
        let fixedValue = timePower;
        if (fixedValue < 0 ) {
            fixedValue *= -1;
        }
        
        const tickDistance = stepSize/10.*kInitPixelsPerSecond*timeZoomFactor;
        const bigTickDistance = stepSize*kInitPixelsPerSecond*timeZoomFactor;
        for( let bigTick = 0; bigTick < end; bigTick +=stepSize) {
            let t = (bigTick-minVisibleTime)*kInitPixelsPerSecond*timeZoomFactor;
            if ((t + bigTickDistance) < 0 ) continue;
            if (t > time_view.width) break;
            graph_context.strokeStyle = "black";
            time_context.beginPath()
            time_context.moveTo(t,0)
            time_context.lineTo(t, time_view.height/2);
            time_context.stroke()
            const sec = bigTick.toFixed(fixedValue);
            
            time_context.fillText(sec.toString()+"s", t, time_view.height*0.75)
            for(let tick = 1; tick < 10; ++tick) {
                let pos = t+tick*tickDistance;
                time_context.beginPath();
                time_context.moveTo(pos,0);
                time_context.lineTo(pos, time_view.height/4);
                time_context.stroke();
            }
        }
    }
    
    function colorToUse(transition) {
        if (!transition.isSrc) {
            if ( ('act' in transition) && transition.act != 2) {
                return activityToColor[transition.act];
            } else {
                return typeToColor[transition.type+kTypeOffset];
            }
        }
        return kSourceColor;
    }
    
    function drawGraph() {
        const scale = kInitPixelsPerSecond * timeZoomFactor;
        const maxVisibleTime = time_view.width/scale+minVisibleTime;
        //console.log(minVisibleTime);
        graph_context.save();
        graph_context.setTransform(1,0,0,1,0,0);
        graph_context.fillStyle = "#D5D6C6"
        graph_context.fillRect(0,0,graph.width,graph.height);
        graph_context.restore();
        
        graph_context.strokeStyle = "black";
        graph_context.scale(1,1);
        
        let offset = graph_vertical_offset;
        for( let grouping of data.transitions) {
            if (offset > graph.height) break;
            for( let slot of grouping.slots ) {
                if (offset > graph.height) break;
                if (offset+kBoxHeight >= 0) {
                    for( let transition of slot) {
                        if (maxVisibleTime < transition.start) {
                            break;
                        }
                        if (minVisibleTime > transition.finish) {
                            continue;
                        }
                        if(transition == selected_item) {
                            graph_context.fillStyle = "white";
                        } else {
                            graph_context.fillStyle = colorToUse(transition);
                        }
                        graph_context.fillRect(scale*(transition.start-minVisibleTime), offset, scale*(transition.finish-transition.start), kBoxHeight);
                    }
                }
                offset += kRowHeight;
            }
        }
        drawNames();
        drawTime();
        
    }
    graph.width = div.clientWidth;
    const kInitPixelsPerSecond = graph.width/kEndTime;
    const max_graph_width = graph.width;
    
    graph.height = div.clientHeight
    name_view.width = left.clientWidth
    name_view.height = left.clientHeight
    time_view.width = bottom.clientWidth
    time_view.height = bottom.clientHeight
    drawGraph()
    
    let graph_isDragging = false;
    let graph_mouseDown = false;
    let graph_mouseDownPosition = {x:0, y:0};
    let graph_dragStartPosition = { x: 0, y: 0 };
    
    let time_isDragging = false;
    let time_dragStartPosition = 0;
    
    function graph_translate(xDiff, yDiff) {
        let original = minVisibleTime;
        minVisibleTime -= xDiff/kInitPixelsPerSecond/timeZoomFactor;
        //console.log(xDiff, original, minVisibleTime);
        const timeEnd = (max_graph_width)/kInitPixelsPerSecond/timeZoomFactor+minVisibleTime;
        if (timeEnd > kEndTime) {
            minVisibleTime = kEndTime - max_graph_width/kInitPixelsPerSecond/timeZoomFactor;
        }
        if (minVisibleTime < 0) {
            minVisibleTime = 0;
        }
        original = graph_vertical_offset;
        graph_vertical_offset += yDiff;
        if (graph_vertical_offset < -max_graph_height) {
            graph_vertical_offset = -max_graph_height;
        }
        if (graph_vertical_offset > 0) {
            graph_vertical_offset = 0;
        }
    }
    
    function getTransformedPoint(x, y) {
        const originalPoint = new DOMPoint(x, y);
        return graph_context.getTransform().invertSelf().transformPoint(originalPoint);
    }
    
    function graph_onMouseDown(event) {
        graph_mouseDown = true;
        graph_mouseDownPosition = {x:event.offsetX, y:event.offsetY};
        graph_dragStartPosition = getTransformedPoint(event.offsetX, event.offsetY);
    }
    
    function graph_onMouseMove(event) {
        let currentTransformedCursor = getTransformedPoint(event.offsetX, event.offsetY);
        
        if (graph_mouseDown) {
            if (Math.abs(graph_mouseDownPosition.x-event.offsetX)> 5 ||
                Math.abs(graph_mouseDownPosition.y-event.offsetY)> 5) {
                graph_isDragging = true;
            }
        }
        if (graph_isDragging) {
            graph_translate(currentTransformedCursor.x - graph_dragStartPosition.x, currentTransformedCursor.y - graph_dragStartPosition.y);
            graph_dragStartPosition.x = currentTransformedCursor.x;
            graph_dragStartPosition.y = currentTransformedCursor.y;
            drawGraph();
        }
    }
    
    function moduleName(id) {
        if (id ==0) {
            return "source";
        }
        return "";
    }
    
    function doUnselection() {
        selected_view.innerHTML = "Selected: [click on box in graph]";
        selected_item = null;
    }
    
    function moduleIdToName(id) {
        if (id < 0) {
            return data.esModules[-1*id];
        }
        return data.modules[id];
    }
    
    function duration(t) {
        if (t < 0.001) {
            return (t*1000000).toFixed()+"us";
        }
        if (t < 0.1) {
            return (t*1000).toFixed(3)+"ms";
        }
        return t.toFixed(6)+"s";
    }
    
    function updateSelectedView(item) {
        if ('isSrc' in item) {
            if (item.isSrc) {
                if(item.mod) {
                    selected_view.innerHTML ="Selected: source reading data product: for module "+item.mod+" "+moduleIdToName(item.mod)+ " while "+activityToName[item.act]+" "+typeToName[item.type+kTypeOffset] +" start: "+item.start.toFixed(6)+"s finish: "+item.finish.toFixed(6)+"s duration: "+duration(item.finish-item.start);

                } else {
                    selected_view.innerHTML = "Selected: source "+typeToName[item.type+kTypeOffset]+" id: "+item.id+" run: "+item.sync[0]
                    +" lumi: "+item.sync[1]+ " event: "+item.sync[2]+" start: "+item.start.toFixed(6)+"s finish:"+item.finish.toFixed(6)+"s duration: "+duration(item.finish-item.start);
                }
            } else {
                selected_view.innerHTML = "Selected: "+typeToName[item.type+kTypeOffset]+" id: "+item.id+" run: "+item.sync[0]
                +" lumi: "+item.sync[1]+ " event: "+item.sync[2]+" start: "+item.start.toFixed(6)+"s finish:"+item.finish.toFixed(6)+"s duration: "+duration(item.finish-item.start);
            }
        } else {
            let transform = '';
            if (item.call != 0 && item.mod > 0) {
                transform = ' transform '
            }
            selected_view.innerHTML ="Selected: module : "+item.mod+" "+moduleIdToName(item.mod)+ " while "+activityToName[item.act]+transform+" "+typeToName[item.type+kTypeOffset] +" start: "+item.start.toFixed(6)+"s finish: "+item.finish.toFixed(6)+"s duration: "+duration(item.finish-item.start);
        }
    }
    function doSelection(items, x) {
        let time = x/kInitPixelsPerSecond/timeZoomFactor+minVisibleTime;
        //console.log(time);
        if( time < 0 || time > kEndTime) {
            doUnselection();
            drawGraph();
            return;
        }
        selected_item = null;
        for( let item of items) {
            if (time < item.start) {
                break;
            }
            if (time > item.start && time < item.finish) {
                selected_item = item;
                if ( overlappingTransitions.includes(selected_item.type)) {
                    continue;
                } else {
                    break;
                }
            }
        }
        if (selected_item) {
            updateSelectedView(selected_item);
        } else {
            doUnselection();
        }
        drawGraph();
    }
    
    function graph_onMouseUp(event) {
        if (graph_mouseDown && ! graph_isDragging) {
            //trying to select
            const selectionPoint = getTransformedPoint(event.offsetX, event.offsetY);
            const vertIndex = Math.floor((selectionPoint.y-graph_vertical_offset)/kRowHeight);
            
            let presentIndex = 0;
            let container = null;
        outer: for(let grouping of data.transitions) {
            for(let slot of grouping.slots) {
                if (presentIndex == vertIndex) {
                    container = slot;
                    break outer;
                }
                ++presentIndex;
            }
        }
            if (!container) {
                doUnselection();
                drawGraph();
            } else {
                doSelection(container, selectionPoint.x);
            }
        }
        graph_isDragging = false;
        graph_mouseDown = false;
    }
    
    function graph_onMouseOut() {
        graph_isDragging = false
        graph_mouseDown = false;
    }
    function graph_onWheel(event) {
        if (event.ctrlKey) {
            let currentTransformedCursor = getTransformedPoint(event.offsetX, event.offsetY);
            const zoom = event.deltaY < 0 ? 1.02 : 0.98;
            const originalScale = 1./timeZoomFactor/kInitPixelsPerSecond;
            timeZoomFactor *= zoom;
            const newScale = 1./timeZoomFactor/kInitPixelsPerSecond;
            //Keep the cursor at the same time and change zoom around that point
            minVisibleTime = minVisibleTime + currentTransformedCursor.x*(originalScale-newScale);
        } else {
            graph_translate(-1*event.deltaX, -1*event.deltaY);
        }
        drawGraph();
        event.preventDefault()
    }

    window.addEventListener('resize',function(){
		graph_context.canvas.width = graph.clientWidth;
		graph_context.canvas.height = graph.clientHeight;
		name_context.canvas.width = name_view.clientWidth;
		name_context.canvas.height = name_view.clientHeight;
		time_context.canvas.width = time_view.clientWidth;
		time_context.canvas.height = time_view.clientHeight;
		drawGraph();
	}, false);
    graph.addEventListener('mousedown', graph_onMouseDown)
    graph.addEventListener('mousemove', graph_onMouseMove)
    graph.addEventListener('mouseup', graph_onMouseUp)
    graph.addEventListener('mouseout', graph_onMouseOut)
    graph.addEventListener('wheel', graph_onWheel);
    
    function time_onMouseDown(event) {
        time_isDragging = true;
        time_dragStartPosition = getTransformedPoint(event.offsetX, 0).x;
    }
    
    function time_onMouseMove(event) {
        let currentTransformedCursor = getTransformedPoint(event.offsetX, 0);
        
        if (time_isDragging) {
            graph_translate(currentTransformedCursor.x - time_dragStartPosition, 0);
            time_dragStartPosition = currentTransformedCursor.x;
            drawGraph();
        }
    }
    
    function time_onMouseUp() {
        time_isDragging = false;
    }
    
    function time_onMouseOut() {
        time_isDragging = false
    }
    
    function time_onWheel(event) {
        if (event.ctrlKey) {
            let currentTransformedCursor = getTransformedPoint(event.offsetX, event.offsetY);
            const zoom = event.deltaY < 0 ? 1.02 : 0.98;
            const originalScale = 1./timeZoomFactor/kInitPixelsPerSecond;
            timeZoomFactor *= zoom;
            const newScale = 1./timeZoomFactor/kInitPixelsPerSecond;
            //Keep the cursor at the same time and change zoom around that point
            minVisibleTime = minVisibleTime + currentTransformedCursor.x*(originalScale-newScale);
        } else {
            graph_translate(-1*event.deltaX, 0);
        }
        //console.log(minVisibleTime);
        drawGraph();
        event.preventDefault();
        
    }
    
    time_view.addEventListener('mousedown', time_onMouseDown)
    time_view.addEventListener('mousemove', time_onMouseMove)
    time_view.addEventListener('mouseup', time_onMouseUp)
    time_view.addEventListener('mouseout', time_onMouseOut)
    time_view.addEventListener('wheel', time_onWheel);
    
    function zoom_in_click(event) {
        const zoom = 1.1;
        const originalScale = 1. / timeZoomFactor / kInitPixelsPerSecond;
        timeZoomFactor *= zoom;
        const newScale = 1. / timeZoomFactor / kInitPixelsPerSecond;
        //Keep the center of the view at the same time and change zoom around that point
        minVisibleTime = minVisibleTime + max_graph_width/2*(originalScale-newScale);
        drawGraph();
    }
    function zoom_out_click(event) {
        const zoom = 0.909;
        const originalScale = 1. / timeZoomFactor / kInitPixelsPerSecond;
        timeZoomFactor *= zoom;
        const newScale = 1. / timeZoomFactor / kInitPixelsPerSecond;
        //Keep the center of the view at the same time and change zoom around that point
        minVisibleTime = minVisibleTime + max_graph_width/2*(originalScale-newScale);
        drawGraph();
    }
    zoom_in_button.addEventListener("click", zoom_in_click);
    zoom_out_button.addEventListener("click", zoom_out_click);

    function name_onWheel(event) {
        let offset = 0;
        graph_translate(0, -1*event.deltaY);
        drawGraph();
        event.preventDefault()
    }
    name_view.addEventListener('wheel', name_onWheel);
}
