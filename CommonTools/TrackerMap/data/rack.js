 var TrackerCrate = {} ;
TrackerCrate.thisFile = "rack.js" ;
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
	    var rack = Math.floor(detid/1000);
	
            
	//    parent.document.getElementById('print2').setAttribute("src",parent.servername+parent.tmapname+"crate"+crate+".html#"+detid);
            parent.window.setip3(parent.servername+parent.tmapname+"psurack"+rack+".html#"+detid);
	     //   alert(detid);
		//alert(top.document.getElementById('print1'));
	    var cmodid = myPoly.getAttribute("cmodid");
            var modules = new Array();
            var listinit = cmodid.indexOf('(')+1; 
            var modlist = cmodid.substring(listinit);
            //var comma = modlist.indexOf(','); 
            //var modulid = modlist.substring(0,comma);
            modules=modlist.split(",");
            var layer = Math.floor(modules[0]/100000);
	    if(layer!=parent.loaded){parent.loaded=layer;parent.remotewin.location.href=parent.servername+parent.tmapname+"layer"+layer+".xml";}
	    //alert(modules.length+" "+modules[0]);
            opacity=0.4;
            for( var imod=0; imod < (modules.length-1);imod++){
            myPoly.setAttribute("style","stroke: black; stroke-width: 1") ;
            if(parent.remotewin.document.getElementById(modules[imod])!=null) {styledef=parent.remotewin.document.getElementById(modules[imod]).getAttribute("style");if(styledef==null||styledef=="")parent.remotewin.document.getElementById(modules[imod]).setAttribute("style"," stroke: black; stroke-width: 1") ; else parent.remotewin.document.getElementById(modules[imod]).setAttribute("style",""); }
     }
     } 
       if (evt.type == "mouseout") {
    var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","cursor:default; fill-opacity: "+opacity) ;

     }


     }

