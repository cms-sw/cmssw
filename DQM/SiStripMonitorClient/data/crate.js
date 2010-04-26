 var TrackerCrate = {} ;
TrackerCrate.thisFile = "crate.js" ;
TrackerCrate.init = function()
 {
  showData = TrackerCrate.showData;
     }
 
TrackerCrate.showData = function (evt) {
    var myPoly = evt.currentTarget;
       if (evt.type == "mouseover") {
    var myPoly = evt.currentTarget;
       var myTracker = myPoly.getAttribute("POS");
          var myTracker1 = "  value="+myPoly.getAttribute("value");
               myTracker1 = myTracker1+"  count="+myPoly.getAttribute("count");
             var textfield=document.getElementById('line1');
        textfield.firstChild.nodeValue=myTracker;
              textfield=document.getElementById('line2');
        textfield.firstChild.nodeValue=myTracker1;
        opacity=0.2;
        myPoly.setAttribute("style","cursor:crosshair; fill-opacity: "+opacity) ;

      //top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var crate = Math.floor(id/1000000);
opacity=0.4;
myPoly.setAttribute("style","fill-opacity: "+opacity+"; stroke: black; stroke-width: 2") ;
	//    parent.document.getElementById('print2').setAttribute("src",parent.servername+parent.tmapname+"crate"+crate+".html#"+detid);
      parent.window.setip1(parent.servername+parent.tmapname+"crate"+crate+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	var cmodid = myPoly.getAttribute("cmodid");
            var modules = new Array();
            
            var layer = Math.floor(cmodid/100000);
	    if(layer!=parent.loaded){parent.loaded=layer;parent.remotewin.location.href=parent.servername+parent.tmapname+"layer"+layer+".xml";}
	    //alert(modules.length+" "+modules[0]);
            opacity=0.4;
            
            myPoly.setAttribute("style","stroke: black; stroke-width: 1") ;
            if(parent.remotewin.document.getElementById(cmodid)!=null){styledef=parent.remotewin.document.getElementById(cmodid).getAttribute("style");
	      if(styledef==null||styledef=="")parent.remotewin.document.getElementById(cmodid).setAttribute("style","stroke: black; stroke-width: 1") ; 
	      else parent.remotewin.document.getElementById(cmodid).setAttribute("style",""); }
         
     }
       if (evt.type == "mouseout") {
    var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","cursor:default; fill-opacity: "+opacity) ;

     }


     }

