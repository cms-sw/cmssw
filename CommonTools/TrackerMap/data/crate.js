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
      top.document.getElementById('currentElementText').setAttribute("value",myTracker);
     }
            if (evt.type == "click") {
	    var detid = myPoly.getAttribute("detid");
	    var id = myPoly.getAttribute("id");
	    var crate = Math.floor(id/100000);
	    top.document.getElementById('print2').setAttribute("src","tmapcrate"+crate+".html#"+detid);
	    //alert(top.document.getElementById('print1'));
	    
     }

     }

