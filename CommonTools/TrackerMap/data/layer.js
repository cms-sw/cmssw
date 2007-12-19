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
      top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var layer = Math.floor(id/100000);
	    top.document.getElementById('print1').setAttribute("src","tiflayer"+layer+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	    
     }

     }

