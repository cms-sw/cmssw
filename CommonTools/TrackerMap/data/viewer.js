// add event listening
function aaManageEvent(eventObj, event, eventHandler) {
   if (eventObj.addEventListener) {
      eventObj.addEventListener(event, eventHandler,false);
   } else if (eventObj.attachEvent) {
      event = "on" + event;
      eventObj.attachEvent(event, eventHandler);
   }
}

// Tabs master
function Tabs (active,inactive) {
   current=0;
   tab = new Array();
   panels = new Array();
   activeColor = active;
   inactiveColor = inactive;

   // add tab
   this.addTab = function (tabItem) {
                          var index = tab.length;
                          tab[index] = tabItem;
                          aaManageEvent(tabItem,"click",function() {
                                 tabs.showPanel(index);});
                          };

   // add panel
   this.addPanel = function (panel) {
                          panels[panels.length] = panel;
                         };

   // display the clicked panel, 'hide the previous'
   this.showPanel = function (index) {
                     panels[current].style.display='none';
                     tab[current].style.backgroundColor=inactiveColor;
                     tab[index].style.backgroundColor=activeColor;
                     panels[index].style.display='block';
                     current = index;
		     if(current==0)makeDraggable('img1'); else makeDraggable('img2');
                     };
   }


// setup tabs and matching panels, assumed in parallel order
function setUpTabs() {
   var divs = document.getElementsByTagName('div');
   var tabCount = 0;
   for (var i = 0; i < divs.length; i++) {
       if (divs[i].className == 'name') {
             tabs.addTab(divs[i]);
        } else if (divs[i].className == 'content') {
             tabs.addPanel(divs[i]);
             divs[i].style.display = 'none';
            
       }
   }
   tabs.showPanel(0);
   document.getElementById('img1').src=tmapname+".png";
   document.getElementById('img2').src=tmapname+"fed.png";
   makeDraggable('img1');makeCliccable('img1');
   setFull('layer1');
}


// create a mouse point
function mousePoint(x,y) {
   this.x = x;
   this.y = y;
}

// find mouse position
function mousePosition(evnt){
  var x = parseInt(evnt.clientX); 
  var y = parseInt(evnt.clientY); 
  return new mousePoint(x,y);
}

// get element's offset position within page 
function getMouseOffset(target, evnt){
   evnt = evnt || window.event;
   var mousePos  = mousePosition(evnt);
   var x = mousePos.x - target.offsetLeft;
   var y = mousePos.y - target.offsetTop;
   return new mousePoint(x,y);
}

// turn off dragging
function mouseUp(evnt){
   dragObject = null;
}

// capture mouse move, only if dragging
function mouseMove(evnt){
   if (!dragObject) return;
   evnt = evnt || window.event;
   var mousePos = mousePosition(evnt);

   // if draggable, set new absolute position
   if(dragObject){
      dragObject.style.position = 'absolute';

      dragObject.style.top      = mousePos.y - mouseOffset.y + "px";
      dragObject.style.left     = mousePos.x - mouseOffset.x + "px";
      return false;
    }
}

// make object cliccable 
function makeCliccable(item){
   if (item) {
      item = document.getElementById(item);
      item.onclick = function(evnt) {
                         if(dragObject)return;
                         evnt = evnt || window.event;
                         var mousePos = mousePosition(evnt);
			  layer = getLayer(mousePos.x-this.offsetLeft,mousePos.y-this.offsetTop);
             	 // alert((mousePos.x-this.offsetLeft)+" "+(mousePos.y-this.offsetTop)+" "+layer);
			  setSingle('layer1',layer);
                         return false; };
   }
}

function getLayer(ix,iy){
iy=iy - 212;
ix=ix - 52;
var add;
var xsize=340;
var ysize=200;
 var res=0;
  if(iy <= xsize){//endcap+z
   add = 15;
    res = Math.floor(ix/ysize);
     res = res+add+1;
    }
   if(iy > xsize && iy< 3*xsize){//barrel
    add=30;
     if(ix < 2*ysize){
        res=1;
         }else {
  res = Math.floor(ix/(2*ysize));
     if(iy < 2*xsize)res=res*2+1; else res=res*2;
      }
       res = res+add;
      }
     if(iy >= 3*xsize){        //endcap-z
      res = Math.floor(ix/ysize);
       res = 15-res;
      }
return res
 }
// make object draggable
function makeDraggable(item){
   tmapObject=item;
   if (item) {
      item = document.getElementById(item);
      item.onmousedown = function(evnt) {
                         dragObject  = this;
                         mouseOffset = getMouseOffset(this, evnt);
                         return false; };
   }
}

function zoomIt(inOrOut) {
 if (inOrOut == "TIB") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-600px";
  imgObject.style.left     =  "-800px";
 return false;
}																			 
}																			 
 if (inOrOut == "TOB") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-600px";
  imgObject.style.left     =  "-1600px";
 return false;
}																			 
}																			 
 if (inOrOut == "TID+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-240px";
  imgObject.style.left     =  "-600px";
 return false;
}																			 
}																			 
 if (inOrOut == "TEC+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-240px";
  imgObject.style.left     =  "-1200px";
 return false;
}																			 
}																			 
 if (inOrOut == "TID-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-1020px";
  imgObject.style.left     =  "-600px";
 return false;
}																			 
}																			 
 if (inOrOut == "TEC-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-1020px";
  imgObject.style.left     =  "-1200px";
 return false;
}																			 
}																			 
 if (inOrOut == "PIXB") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-600px";
  imgObject.style.left     =  "0px";
 return false;
}																			 
}																			 
 if (inOrOut == "FPIX-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-1020px";
  imgObject.style.left     =  "0px";
 return false;
}																			 
}																			 
 if (inOrOut == "FPIX+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-240px";
  imgObject.style.left     =  "0px";
 return false;
}																			 
}																			 
	if (inOrOut == "SVG") {if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',layer);return false;}																 
	if (inOrOut == "Home") {if(tmapObject=='img1')setFull('layer1');else setFull('layer2');return false;}																 
	if (inOrOut == "<") {layer=layer-1;if(layer==0)layer=43;if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',layer);return false;}																 
	if (inOrOut == ">") {;layer=layer+1;if(layer==44)layer=1;if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',layer);return false;}																 
}																			 
function setSingle(elemento,layer1){
 if(elemento){
   frame = document.getElementById(elemento);
   if(tmapObject=='img1')divObject = document.getElementById('div1');else divObject = document.getElementById('div2');
   if(tmapObject=='img1')printObject = document.getElementById('print1');else printObject = document.getElementById('print2');
   if(frame.style.display=='none'){
      frame.style.display='';
      printObject.style.display='';
      divObject.style.display='none';
         frame.src=tmapname+"layer"+layer1+".xml";
         printObject.src=tmapname+"layer"+layer1+".html";
	    } else {
         frame.src=tmapname+"layer"+layer1+".xml";
         printObject.src=tmapname+"layer"+layer1+".html";
	           }
		   }
		   }
function setFull(elemento){
 if(elemento){
   frame = document.getElementById(elemento);
   if(tmapObject=='img1')divObject = document.getElementById('div1');else divObject = document.getElementById('div2');
   if(tmapObject=='img1')printObject = document.getElementById('print1');else printObject = document.getElementById('print2');
   if(frame.style.display=='none'){
	    } else {
	       frame.style.display='none';
	       printObject.style.display='none';
               divObject.style.display='';
	           }
		   }
		   }
function toggle(elemento){
 if(elemento){
   frame = document.getElementById(elemento);
   if(tmapObject=='img1')divObject = document.getElementById('div1');else divObject = document.getElementById('div2');
   if(tmapObject=='img1')printObject = document.getElementById('print1');else printObject = document.getElementById('print2');
   if(frame.style.display=='none'){
      frame.style.display='';
      printObject.style.display='';
      divObject.style.display='none';
         frame.src="tmaplayer"+layer+".xml";
         printObject.src="tmaplayer"+layer+".html";
	    } else {
	       frame.style.display='none';
	       printObject.style.display='none';
               divObject.style.display='';
	           }
		   }
		   }
