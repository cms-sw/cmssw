CommonActions = {} ;

//___________________________________________________________________________________
// Get File
CommonActions.GetFile = function() {                  // Unused?
  var obj = document.getElementById("filename");
  var fname =  obj.options[obj.selectedIndex].value;
  return fname;
}

//___________________________________________________________________________________
// Get Reference File
CommonActions.GetRefFile = function() {               // Unused? 
  var obj = document.getElementById("ref_filename");
  var fname = obj.options[obj.selectedIndex].value;
  return fname;
}

//___________________________________________________________________________________
// Get Selected Histos
CommonActions.GetSelectedHistos = function() {
  var hlist = new Array();
  var obj = document.getElementById("histolistarea");
  var len = obj.length; 
  if (len == 0) {
    alert("Histogram List Area Empty!");
  } else {
    for (var i = 0; i < len; i++) {
      if (obj.options[i].selected) {
	hlist[hlist.length] = obj.options[i].value;
      }
    }
  }
  return hlist;
}

//___________________________________________________________________________________
CommonActions.GetPlotOptions = function() {        // Unused?
  var hlist = new Array();
  var obj = document.getElementById("histolistarea");
  var len = obj.length; 
  if (len == 0) {
    alert("Histogram List Area Empty!");
  } else {
    for (var i = 0; i < len; i++) {
      if (obj.options[i].selected) {
	hlist[hlist.length] = obj.options[i].value;
      }
    }
  }
  return hlist;
}
