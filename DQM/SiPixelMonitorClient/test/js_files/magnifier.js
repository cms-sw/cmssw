// Credit: original code by Mihai Parparita (mihai@persistent.info)
// Improved and customized by D. Menasce (http://hal9000.mib.infn.it/~menasce)
// This script is now a class with its own namespace to avoid clashes if
// used in a context where other JavaScript codes are present
  
DLMLens.prototype.kShadowPadding	= 17;
DLMLens.prototype.kDefaultMagnifierSize = 0; // index into the arrays below
DLMLens.prototype.kMagnifierSizes       = new Array(0, 100, 150, 300);
DLMLens.prototype.kMagnifierSizeNames   = new Array('off', 'small', 'medium', 'large');
DLMLens.prototype.kControllerPrefix     = '<Font Color="#fceeb8">Lens:</Font>&nbsp;';
DLMLens.prototype.shadowBaseUrl         = "http://hal9000.mib.infn.it/~menasce/PBaseStuff/shadow" ;

//====================================================================================
function DLMLens(baseID, zoomedURL, zoomedWidth, zoomedHeight)
{
 this.MagnifierPosition         = MagnifierPosition;
 this.ControllerSizeButtonClick = ControllerSizeButtonClick;
 this.MagnifierResize           = MagnifierResize;
 this.MagnifierMouseDown        = MagnifierMouseDown;
 this.MagnifierMouseUp          = MagnifierMouseUp;
 this.MagnifierDrag             = MagnifierDrag;
 this.loadMagnifier             = loadMagnifier;
 this.update                    = update;
 this.baseID                    = baseID ;
 this.mssg ;
 
 this.loadMagnifier             = loadMagnifier ;
 this.loadMagnifier(baseID, zoomedURL, zoomedWidth, zoomedHeight) ;
}     

//====================================================================================
function update(baseID, zoomedURL, zoomedWidth, zoomedHeight)
{
 this.MagnifierPosition         = MagnifierPosition;
 this.ControllerSizeButtonClick = ControllerSizeButtonClick;
 this.MagnifierResize           = MagnifierResize;
 this.MagnifierMouseDown        = MagnifierMouseDown;
 this.MagnifierMouseUp          = MagnifierMouseUp;
 this.MagnifierDrag             = MagnifierDrag;
 this.loadMagnifier             = loadMagnifier;
 this.baseID                    = baseID ;
 this.mssg ;
 
 this.loadMagnifier             = loadMagnifier ;
 this.loadMagnifier(baseID, zoomedURL, zoomedWidth, zoomedHeight) ;
}
//====================================================================================
function MagnifierPosition()
{     
 this.style.left 	= Math.round(this.xPosition - 1 - this.size/2) + "px";
 this.style.top  	= Math.round(this.yPosition - 1 - this.size/2) + "px";

 this.shadow.style.left = Math.round(this.xPosition - this.size/2 - DLMLens.prototype.kShadowPadding) + "px";
 this.shadow.style.top  = Math.round(this.yPosition - this.size/2 - DLMLens.prototype.kShadowPadding) + "px";

 var magnifierCenterX	= Math.round(this.xPosition * this.xMultiplier - this.size/2);
 var magnifierCenterY	= Math.round(this.yPosition * this.yMultiplier - this.size/2);

 var baseVect 		= document.getElementsByName(this.baseID);
 var base               = baseVect[0]
 var px   		= findPosX(base) ;
 var py   		= findPosY(base) ;
 var xOff 		= Math.round(-magnifierCenterX + px * this.xMultiplier) ;
 var yOff 		= Math.round(-magnifierCenterY + py * this.yMultiplier) ;
 this.style.backgroundPosition = xOff + "px " + yOff + "px";
}

//====================================================================================
function ControllerSizeButtonClick(event)
{
 if (!event) event = window.event;
 var button = event.currentTarget || event.srcElement;
 button.parentNode.magnifier.resize(button.magnifierSize);
}

//====================================================================================
function MagnifierResize(size)
{
 this.size = DLMLens.prototype.kMagnifierSizes[size];

 for (var i=0; i < this.controller.sizeButtons.length; i++)
 {
  if (i == size)
          this.controller.sizeButtons[i].className = "magnifierControllerButtonSelected";
  else
          this.controller.sizeButtons[i].className = "magnifierControllerButton";
 }

 if (this.size == 0)
 {
  this.shadow.style.display = "none";
  this.style.display        = "none";
 } else {
  var shadow     = this.shadow;
  var shadowSize = this.size + 2 * DLMLens.prototype.kShadowPadding;

  // MSIE 5.x/6.x must be treated specially in order to make them use the PNG alpha channel
  var shadowImageSrc = DLMLens.prototype.shadowBaseUrl + size + ".png";
  if (shadow.runtimeStyle)
  {
    shadow.style.filter = "progid:DXImageTransform.Microsoft.AlphaImageLoader(src='" +
        		   shadowImageSrc +
        		  "', sizingMethod='scale')";
    alert(shadow.style.filter) ;
//	shadow.runtimeStyle.filter = "progid:DXImageTransform.Microsoft.Alpha(src='" +
//				      shadowImageSrc +
//				     "', sizingMethod='scale')";
  } else {
    shadow.style.backgroundImage = "url(" + shadowImageSrc + ")";
  }
  shadow.style.width   = shadowSize + "px";
  shadow.style.height  = shadowSize + "px";
  shadow.style.display = "block";

  if (this.runtimeStyle) // msie counts the border as being part of the width
          this.size += 2; // must compensate

  this.style.width   = this.size + "px";
  this.style.height  = this.size + "px";
  this.style.display = "block";
  this.position();
 }
}

//====================================================================================
function MagnifierMouseDown(event)
{
 if (!event) event = window.event;

 document.body.magnifier = this;
 this.inDrag = true;
 if (event.pageX)
 {
  this.startX = event.pageX;
  this.startY = event.pageY;
 }
 else if (event.clientX)
 {
  this.startX = event.clientX;
  this.startY = event.clientY;
 }
 else
 {
  alert("don't know how to get position out of event");
  return;
 }
 this.savedCursor  = this.style.cursor;
 this.style.cursor = "crosshair";
}

//====================================================================================
function MagnifierMouseUp()
{
 if (this.inDrag)
 {
  this.inDrag		  = false;
  this.style.cursor	  = this.savedCursor;
  document.body.magnifier = null;
 }
}

//====================================================================================
function MagnifierDrag(event)
{
 if (!event) event = window.event;
 var magnifier = this.magnifier; // we're actually in the body's onmousemove handler

 if (magnifier && magnifier.inDrag)
 {
  var eventX;
  var eventY;

  if (event.pageX)
  {
   eventX = event.pageX;
   eventY = event.pageY;
  }
  else if (event.clientX)
  {
   eventX = event.clientX;
   eventY = event.clientY;
  }
  else
  {
   return;
  }

  magnifier.xPosition += eventX - magnifier.startX
  magnifier.yPosition += eventY - magnifier.startY;
   
  magnifier.startX = eventX;
  magnifier.startY = eventY;

  magnifier.position();
 }
}

//====================================================================================
function loadMagnifier(baseID, zoomedURL, zoomedWidth, zoomedHeight)
{
 //DM_TraceWindow(thisFile,arguments.callee.name,"zoomedURL: "+zoomedURL) ;  
 var baseVect	    = document.getElementsByName(baseID);
 if( baseVect.length < 1 )
 {
  alert("[magnifier.js::loadMagnifier] No item named "+baseID) ;
 }
 if( baseVect.length > 1 )
 {
  alert("[magnifier.js::loadMagnifier] More than one item named "+baseID) ;
 }
 var base           = baseVect[0] ;
 var zoomedImage    = document.getElementsByName("zoomedImage") ;
 var checkItem      = null ;

 if( !zoomedImage )
 {
  zoomedImage 	    = document.createElement("img");
  zoomedImage.name  = "zoomedImage" ;
  zoomedImage.src   = zoomedURL;		    
  
 } else {
  checkItem         = document.getElementById(this.baseID + "Magnifier") ;
  if (!checkItem )
  {
   this.magnifier   = document.createElement("div");
  } else {
   this.magnifier   = checkItem  ;
  }
 }
 
 // get the regular image
 var normalImage = null;
 for (var i=0; i < base.childNodes.length; i++)
 {
  if (base.childNodes[i].tagName && base.childNodes[i].tagName.toLowerCase() == "img")
  {
   normalImage = base.childNodes[i];
   break;
  }
 }

 if (normalImage == null)
 {
  alert("[loadMagnifyer] Couldn't find normal image for magnifier " + this.baseID);
  return;
 }

 this.magnifier.xMultiplier    = zoomedWidth /normalImage.width;
 this.magnifier.yMultiplier    = zoomedHeight/normalImage.height;

 var px = findPosX(base) ;
 var py = findPosY(base) ;
 this.magnifier.size           = DLMLens.prototype.kMagnifierSizes[DLMLens.prototype.kDefaultMagnifierSize];
 this.magnifier.xPosition      = normalImage.width  - this.magnifier.size/2 + px;
 this.magnifier.yPosition      = normalImage.height - this.magnifier.size/2 + py;
 this.magnifier.id	       = this.baseID + "Magnifier";
 this.magnifier.className      = "magnifier";

 // styles (only dynamic ones, rest are part of the class)
 this.magnifier.style.backgroundImage = "url(" + zoomedURL + ")";

 // functions
 this.magnifier.onmousedown    = this.MagnifierMouseDown;
 this.magnifier.onmouseup      = this.MagnifierMouseUp;
 document.body.onmousemove     = this.MagnifierDrag; // we attach this handler to the body because if the user moves
        		        		     // the mouse fast enough, it'll go outside the boundaries of the
        		        		     // magnifier, and then the magnifier's mousemove handler won't fire
    
 this.magnifier.position       = this.MagnifierPosition;
 this.magnifier.resize	       = this.MagnifierResize;
 this.magnifier.baseID	       = this.baseID;

 // controller
 checkItem                     = document.getElementById(this.baseID + "MagnifierController") ;
 if( !checkItem )
 {
  this.controller              = document.createElement("span");
 } else {
  this.controller              = checkItem;
 }

 this.controller.id  	       = this.baseID + "MagnifierController";
 this.controller.className     = "magnifierController";

 var controllerPrefix	       = null;
 checkItem                     = document.getElementById(this.baseID + "ControllerPrefix") ;
 if( !checkItem )
 {
  controllerPrefix	       = document.createElement("span");
 } else {
  controllerPrefix	       = checkItem ;
 }
 controllerPrefix.id           = this.baseID + "ControllerPrefix" ;
 controllerPrefix.innerHTML    = DLMLens.prototype.kControllerPrefix;

 controllerPrefix.className    = "magnifierControllerPrefix";
 this.controller.appendChild(controllerPrefix);

 this.controller.sizeButtons   = new Array(DLMLens.prototype.kMagnifierSizes.length);

 for (var i=0; i < DLMLens.prototype.kMagnifierSizes.length; i++)
 {
  var button	               = null;
  checkItem                    = document.getElementById(this.baseID + "Button" + i) ;
  if( !checkItem )
  {
   button	               = document.createElement("span");
  } else {
   button	               = checkItem;
  }
  button.innerHTML             = DLMLens.prototype.kMagnifierSizeNames[i];
  button.id                    = this.baseID + "Button" + i;
  button.className             = "magnifierControllerButton";
  button.onclick               = this.ControllerSizeButtonClick;
  button.magnifierSize         = i;

  this.controller.sizeButtons[i] = button;
  this.controller.appendChild(button);
 }

 // shadow
 checkItem                     = document.getElementById(this.baseID + "MagnifierShadow");
 if( !checkItem )
 { 
  this.shadow	               = document.createElement("div");
 } else {
  this.shadow	               = checkItem;
 }
 this.shadow.id	               = this.baseID + "MagnifierShadow";
 this.shadow.className         = "magnifierShadow";

 this.magnifier.controller     = this.controller; // point objects at each other
 this.controller.magnifier     = this.magnifier;
 this.magnifier.shadow         = this.shadow;

 // add to document and lay out

 var controllerContainer       = null ;
 checkItem                     = document.getElementById(this.baseID + "ControllerContainer") ;
 if (!checkItem )
 {
  controllerContainer          = document.createElement("div");
 } else {
  controllerContainer          = checkItem;
 }
 controllerContainer.id        = this.baseID + "ControllerContainer" ;
 controllerContainer.className = "magnifierControllerContainer";

 controllerContainer.appendChild(this.controller);

 base.insertBefore(controllerContainer, document.getElementById("message"));
 base.appendChild(this.shadow);
 base.appendChild(this.magnifier);
 this.magnifier.resize(DLMLens.prototype.kDefaultMagnifierSize);
}

//====================================================================================
function findPosX(obj)
{
 // credit Peter-Paul Koch & Alex Tingle
 // http://www.quirksmode.org/js/findpos.html
 var curleft = 0;
 if(obj.offsetParent)
 {
   while(1) 
   {
     curleft += obj.offsetLeft;
     if(!obj.offsetParent)
       break;
     obj = obj.offsetParent;
   }
 } else if(obj.x) {
  curleft += obj.x;
 }
 return curleft;
}

//====================================================================================
function findPosY(obj)
{
 // credit Peter-Paul Koch & Alex Tingle 
 // http://www.quirksmode.org/js/findpos.html
 var curtop = 0;
 if(obj.offsetParent)
 {
   while(1)
   {
     curtop += obj.offsetTop;
     if(!obj.offsetParent)
       break;
     obj = obj.offsetParent;
   }
 } else if(obj.y) {
  curtop += obj.y;
 }
 return curtop;
}

