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
          myTracker = myTracker+"  value="+myPoly.getAttribute("value");
               myTracker = myTracker+"  count="+myPoly.getAttribute("count");
             var textfield=document.getElementById('currentElementText');
        textfield.firstChild.nodeValue=myTracker;
        opacity=0.2;
        myPoly.setAttribute("style","cursor:crosshair; fill-opacity: "+opacity) ;

      top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var crate = Math.floor(id/1000000);
opacity=0.4;
myPoly.setAttribute("style","fill-opacity: "+opacity+"; stroke: black; stroke-width: 2") ;
	    top.document.getElementById('print2').setAttribute("src",top.tmapname+"crate"+crate+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	    
     }
       if (evt.type == "mouseout") {
    var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","cursor:default; fill-opacity: "+opacity) ;

     }


     }

