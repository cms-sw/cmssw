//------------------------------------------------------------------------------
// Author: D. Menasce (http://hal9000.mib.infn.it/~menasce)
//
// Script to handle output on a textarea. Style is defined in the appropriate
// css/trace.css stylesheet file. 
//
// To place a suitable textarea on screen, paste the following
// statements in the appropriate place in your HTML document:
//
// <form action="#">
//  <textarea id     ="traceRegion" 
//  	      style  ="font-size: 8pt"
//  	      name   ="abstract" 
//  	      rows   ="16" 
//  	      cols   ="68" 
//  	      bgcolor="#ee1100"></textarea>
//  <br>
//  <button   name   ="clearTextArea" 
//  	      value  ="Clear"
//  	      onClick="clearTrace()"
//  	      style  ="font-size: 8pt; color: #ff5500">Clear</button>
// </form>    
//
//==============================================================================
// Static stuff

var mssg = null ;
  
//==============================================================================
function initializeTrace()
{
 mssg = document.getElementById("traceRegion") ;
 if( !mssg )
 {
  alert("[trace] - Warning: no traceRegion defined\n(See source of css/trace.js)") ;
 }
}

//==============================================================================
function trace(msg)
{
 if( !mssg )
 {
   initializeTrace() ;
 }
 var prevContent = mssg.innerHTML ;
 prevContent    += "\n" ;
 prevContent    += msg ;
 mssg.innerHTML  = prevContent ;
 mssg.scrollTop  = mssg.scrollHeight ; // Bring last line into view
/* This stuff is obsolete masturbation...
 var list	 = prevContent.split('\n') ;
 var newContent  = new Array();
 for( var i=0; i<list.length; i++)
 {
  var nLine = list[i].replace(",", "", "gi") ;
  newContent.push(nLine) ;
  newContent.push("\n") ;
 }
 newContent.push(msg) ;
 mssg.innerHTML = newContent ;
*/
}

//==============================================================================
function clearTrace()
{
 mssg.innerHTML = "" ;
 //alert('[trace] - Message box cleared') ;
}
