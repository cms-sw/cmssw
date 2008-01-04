 var TrackerLayer = {} ;
TrackerLayer.thisFile = "layer.js" ;
TrackerLayer.init = function()
 {
  showData = TrackerLayer.showData;
     }
 
TrackerLayer.showData = function (evt) {
    var myPoly = evt.currentTarget;
       if (evt.type == "mouseover") {
    var myPoly = evt.currentTarget;
       var myTracker = myPoly.getAttribute("POS");
          myTracker = myTracker+"  value="+myPoly.getAttribute("value");
               myTracker = myTracker+"  count="+myPoly.getAttribute("count");
             var textfield=document.getElementById('currentElementText');
        textfield.firstChild.nodeValue=myTracker;
        opacity=0.2;
        myPoly.setAttribute("style","fill-opacity: "+opacity) ;
      top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var layer = Math.floor(id/100000);
            opacity=0.4;
myPoly.setAttribute("style","fill-opacity: "+opacity+"; stroke: black; stroke-width: 2") ;
	    top.document.getElementById('print1').setAttribute("src",top.tmapname+"layer"+layer+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	    
     }
       if (evt.type == "mouseout") {
    var myPoly = evt.currentTarget;
        opacity=1;
        myPoly.setAttribute("style","fill-opacity: "+opacity) ;

     }
     }

