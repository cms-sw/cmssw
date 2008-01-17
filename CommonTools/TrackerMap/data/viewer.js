
// Tabs master
function Tabs (active,inactive) {
   current=0;
   tab = new Array();
   panels = new Array();
   zoomAmount = new Array();
   single = new Array();
   activeColor = active;
   inactiveColor = inactive;

   // add tab
   this.addTab = function (tabItem) {
                          var index = tab.length;
                          tab[index] = tabItem;
			  zoomAmount[index]=1.05;
			  single[index]=false;
                          tabItem.addEventListener("click",function() {
                                 tabs.showPanel(index);},false);
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
		     if(current==0){makeDraggable('img1');tmapObject="img1";} else {makeDraggable('img2');tmapObject="img2";}
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
   zoomFactor=.9;
   document.getElementById('img1').src=tmapname+".png";
   document.getElementById('img2').src=tmapname+"fed.png";
   makeDraggable('img1');
   tmapObject="img1";
   setFull('layer2');
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
			  if(item=="img1"){layer = getLayer(mousePos.x-this.offsetLeft,mousePos.y-this.offsetTop);
			  setSingle('layer1',layer);}else{
                          crate=1;
                          setSingle('layer2',crate);
                             }
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
       layer=34;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-350px";
  imgObject.style.left     =  "-400px";
	 imageWidth=2000;
	 imageHeight=1030;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "TOB") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=38;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-350px";
  imgObject.style.left     =  "-1000px";
	 imageWidth=2000;
	 imageHeight=1030;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "TID+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=19;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-240px";
  imgObject.style.left     =  "-600px";
	 imageWidth=3000;
	 imageHeight=1600;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "TEC+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=22;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-120px";
  imgObject.style.left     =  "-600px";
	 imageWidth=1500;
	 imageHeight=800;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "TID-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=10;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-1120px";
  imgObject.style.left     =  "-600px";
	 imageWidth=3000;
	 imageHeight=1600;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "TEC-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=1;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-510px";
  imgObject.style.left     =  "-600px";
	 imageWidth=1500;
	 imageHeight=800;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "PIXB") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=31;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-600px";
  imgObject.style.left     =  "0px";
	 imageWidth=3000;
	 imageHeight=1600;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "FPIX-z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=14;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-1020px";
  imgObject.style.left     =  "0px";
	 imageWidth=3000;
	 imageHeight=1600;
	 displayImage();
	 }
 return false;
}																			 
}																			 
 if (inOrOut == "FPIX+z") {
   imgObject = document.getElementById('img1');
    if(imgObject){
       layer=16;
       if(single[current])setSingle('layer1',layer); else {
       imgObject.style.position = 'absolute';
   imgObject.style.top      =  "-240px";
  imgObject.style.left     =  "0px";
	 imageWidth=3000;
	 imageHeight=1600;
	 displayImage();
	 }
 return false;
}																			 
}																			 
	if (inOrOut == "SVG") {single[current]=true;if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',crate);return false;}																 
	if (inOrOut == "Full") {single[current]=false;if(tmapObject=='img1')setFull('layer1');else setFull('layer2');return false;}																 
	if (inOrOut == "<") {
        if(tmapObject=='img1'){layer=layer-1;if(layer==0)layer=43;}
        else{crate=crate-1;if(crate==0)crate=ncrates;}
if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',crate);return false;}																 
	if (inOrOut == ">") {
        if(tmapObject=='img1'){layer=layer+1;if(layer==44)layer=1;}
        else{crate=crate+1;if(crate==ncrates)crate=1;}
if(tmapObject=='img1')setSingle('layer1',layer);else setSingle('layer2',crate);return false;}																 
	if (!single[current]&&inOrOut == "Home") {
	 zoomAmount[current]=1.05;
	 imageWidth=1000;
	 imageHeight=550;
   document.getElementById(tmapObject).style.top      =  "-50px";
  document.getElementById(tmapObject).style.left     =  "0px";
	 displayImage();
	  }
	if (!single[current]&&inOrOut == "+") {
	 imageWidth=document.getElementById(tmapObject).getAttribute('width');
	 imageHeight=document.getElementById(tmapObject).getAttribute('height');
	 imageWidth=Math.floor(imageWidth*zoomAmount[current]);
	 imageHeight=Math.floor(imageHeight*zoomAmount[current]);
	 displayImage();
	  }
	if (!single[current]&&inOrOut == "-") {
	 imageWidth=document.getElementById(tmapObject).getAttribute('width');
	 imageHeight=document.getElementById(tmapObject).getAttribute('height');
	 imageWidth=Math.floor(imageWidth/zoomAmount[current]);
	 imageHeight=Math.floor(imageHeight/zoomAmount[current]);
	 displayImage();
	}
}																			 
function displayImage(){
	 var oldimage=document.getElementById(tmapObject);
	 var newimage=oldimage.cloneNode(false);
	 newimage.width=imageWidth;
	 newimage.height=imageHeight;
	 oldimage.parentNode.replaceChild(newimage,oldimage);
	 makeDraggable(tmapObject);
	 if(dragObject)dragObject=newimage;

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
	    } else { }
         if(tmapObject=='img1')frame.src=tmapname+"layer"+layer1+".xml";
         else frame.src=tmapname+"crate"+layer1+".xml";
         if(tmapObject=='img1')printObject.src=tmapname+"layer"+layer1+".html";
         else printObject.src=tmapname+"crate"+layer1+".html";
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
               divObject.style.cursor='hand';
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
