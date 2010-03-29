 var TrackerLayer = {} ;
TrackerLayer.thisFile = "layer.js" ;
TrackerLayer.init = function()
 {
  showData = TrackerLayer.showData;
     }
 
TrackerLayer.showData = function (evt) {
    var myPoly = evt.currentTarget;
       if (evt.type == "mouseover") {
       var myTracker = myPoly.getAttribute("POS");
       var myMessage = myPoly.getAttribute("MESSAGE");
       var separator = myTracker.indexOf("connected");
       var myTracker2 = myTracker.substring(separator);
       myTracker = myTracker.substring(0,separator);
       var myTracker1 = "  value="+myPoly.getAttribute("value");
       myTracker1 = myTracker1+"  count="+myPoly.getAttribute("count");
       var textfield=document.getElementById('line1');
       textfield.firstChild.nodeValue=myTracker;
       textfield=document.getElementById('line3');
       textfield.firstChild.nodeValue=myTracker1;
       textfield=document.getElementById('line2');
       textfield.firstChild.nodeValue=myTracker2;
       textfield=document.getElementById('line4');
       textfield.firstChild.nodeValue=myMessage;
        opacity=0.2;
        myPoly.setAttribute("style","cursor:crosshair; fill-opacity: "+opacity) ;
      //top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var layer = Math.floor(id/100000);
	    var capvids = myPoly.getAttribute("capvids");
	    var comma = capvids.indexOf(',');
	    var apvaddr = parseInt(capvids.substring(1,comma));
	    var crate = Math.floor(apvaddr/1000000);
	    var apvaddr1 = capvids.substring(1,comma);
	    var rest = capvids.substring(comma+1);
	     comma = rest.indexOf(',');if(comma==-1)comma=rest.indexOf(')');
	    var apvaddr2 = rest.substring(0,comma);
	    rest = rest.substring(comma+1);
	    var apvaddr3 = "";
	    if (rest.length>5){comma=rest.indexOf(')');apvaddr3 = rest.substring(0,comma);}
	    //alert(apvaddr1+" "+apvaddr2+" "+apvaddr3);
	    if(crate!=parent.loaded){parent.loaded=crate;parent.remotewin.location.href=parent.servername+parent.tmapname+"crate"+crate+".xml";}
            opacity=0.4;
            myPoly.setAttribute("style","stroke: black; stroke-width: 1") ;
	    if(parent.remotewin.document.getElementById(apvaddr1)!=null) {styledef=parent.remotewin.document.getElementById(apvaddr1).getAttribute("style");if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr1).setAttribute("style"," stroke: black; stroke-width: 1") ; else parent.remotewin.document.getElementById(apvaddr1).setAttribute("style",""); }
	    if(apvaddr2!=""&&parent.remotewin.document.getElementById(apvaddr2)!=null) {styledef=parent.remotewin.document.getElementById(apvaddr2).getAttribute("style");if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr2).setAttribute("style"," stroke: black; stroke-width: 1") ; else parent.remotewin.document.getElementById(apvaddr2).setAttribute("style",""); }
	    if(apvaddr3!=""&&parent.remotewin.document.getElementById(apvaddr3)!=null) {styledef=parent.remotewin.document.getElementById(apvaddr3).getAttribute("style");if(styledef==null||styledef=="")parent.remotewin.document.getElementById(apvaddr3).setAttribute("style"," stroke: black; stroke-width: 1") ; else parent.remotewin.document.getElementById(apvaddr3).setAttribute("style",""); }
	    //parent.document.getElementById('print1').setAttribute("src",parent.servername+parent.tmapname+"layer"+layer+".html#"+detid);
	    parent.window.setip(parent.servername+parent.tmapname+"layer"+layer+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	    
     }
       if (evt.type == "mouseout") {
    var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","cursor:default; fill-opacity: "+opacity) ;

     }
     }

