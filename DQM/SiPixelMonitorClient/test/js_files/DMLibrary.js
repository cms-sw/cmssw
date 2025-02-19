//=========================================================================
// Author: D. Menasce (INFN Milano Bicocca)
// Site: http://hal9000.mib.infn.it/~menasce
//
// This utility creates a window containing a text area that is used for
// trace-back blackboard by other scripts.
//=========================================================================

function DM_TraceWindow(fileCaller,functionCaller,msg)
{
 var theTtop   = 780 ;
 var theLeft   = 500 ;
 var theWidth  = 800 ;
 var theHeight = 350 ;
 
 var theTraceWindow = parent.theTraceWindow ;
 
 if( !theTraceWindow )
 {
  theTraceWindow = DM_OpenWindow(theTtop,theLeft,theWidth,theHeight) ; 
 } else if( theTraceWindow.closed ) {
  theTraceWindow = DM_OpenWindow(theTtop,theLeft,theWidth,theHeight) ; 
 }
 
 if( typeof msg == "undefined" ) msg = "" ;

 var newMsg = DM_FormatMsg(fileCaller,functionCaller,msg) ;
 
 DM_WriteWindow(theTraceWindow, newMsg) ;
 
 parent.theTraceWindow   = theTraceWindow ;
 
 theTraceWindow.onresize = resizer ;
}

//=========================================================================
function resizer()
{
}
//=========================================================================
function DM_OpenWindow(Top,Left,Width,Height)
{  
  theTraceWindow = window.open("", 
  			       "theTraceWindow", 
  			       "menubar   = no,  " +
			       "location  = no,  " +
			       "resizable = no,  " +
			       "scrollbars= yes, " +
			       "titlebar  = yes, " +
			       "status    = yes, " +
			       "left      = "      + Left   + ", " +
			       "top       = "      + Top    + ", " +
			       "height    = "      + Height + ", " +
			       "width     = "      + Width );

  theTraceWindow.document.write("<html>                                                      ");
  theTraceWindow.document.write(" <head>                                                     ");
  theTraceWindow.document.write("  <style type=text/css>                                     "); 
  theTraceWindow.document.write("   .traceRegion                                             ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    font-size:       8pt    ;                               ");
  theTraceWindow.document.write("    background-color:#fdf2d0;                               ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   .button                                                  ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    font-size:       8pt    ;                               ");
  theTraceWindow.document.write("    color:           #ff2222;                               ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   h1                                                       ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    font-size:       9pt    ;                               ");
  theTraceWindow.document.write("    color:           #73d0fb;                               ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("  </style>                                                  ");
  theTraceWindow.document.write("  <script type=text/javascript>                             ");
  theTraceWindow.document.write("   function increaseTraceRegion()                           ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    widerTraceRegion() ;                                    ");
  theTraceWindow.document.write("    higherTraceRegion() ;                                   ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   function clearTraceRegion()                              ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    var theTracer= document.getElementById('traceRegion');  ");
  theTraceWindow.document.write("    theTracer.innerHTML = '';                               ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   function widerTraceRegion()                              ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    var theTracer= document.getElementById('traceRegion');  ");
  theTraceWindow.document.write("    var cols = theTracer.getAttribute('cols');              ");
  theTraceWindow.document.write("    var oldCol = parseInt(cols) + 5;                        ");
  theTraceWindow.document.write("    theTracer.setAttribute('cols',oldCol);                  ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   function higherTraceRegion()                             ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    var theTracer= document.getElementById('traceRegion');  ");
  theTraceWindow.document.write("    var rows = theTracer.getAttribute('rows');              ");
  theTraceWindow.document.write("    var oldRow = parseInt(rows) + 5;                        ");
  theTraceWindow.document.write("    theTracer.setAttribute('rows',oldRow);                  ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("   function defaultTraceRegion()                            ");
  theTraceWindow.document.write("   {                                                        ");
  theTraceWindow.document.write("    var theTracer= document.getElementById('traceRegion');  ");
  theTraceWindow.document.write("    theTracer.setAttribute('rows',19);                      ");
  theTraceWindow.document.write("    theTracer.setAttribute('cols',125);                     ");
  theTraceWindow.document.write("   }                                                        ");
  theTraceWindow.document.write("  </script>                                                 ");
  theTraceWindow.document.write("  <title>                                                   ");
  theTraceWindow.document.write("   Trace back utility                                       ");
  theTraceWindow.document.write("  </title>                                                  ");
  theTraceWindow.document.write(" </head>                                                    ");
  theTraceWindow.document.write(" <body bgcolor=#414141>                                     ");
  theTraceWindow.document.write("  <form action='javascript:void()'>		     	     ");
  theTraceWindow.document.write("   <h1>Trace back region                                    ");
  theTraceWindow.document.write("    <button   name    = 'clearTextArea'	    	     ");
  theTraceWindow.document.write("    	       onclick = 'clearTraceRegion()'	    	     ");
  theTraceWindow.document.write("    	       class   = 'button'		    	     ");
  theTraceWindow.document.write("    	       value   = 'Clear'>		    	     ");
  theTraceWindow.document.write("     Clear					    	     ");
  theTraceWindow.document.write("    </button>  				    	     ");
  theTraceWindow.document.write("    <button   name    = 'moreColumns'  	    	     ");
  theTraceWindow.document.write("    	       onclick = 'widerTraceRegion()'	    	     ");
  theTraceWindow.document.write("    	       class   = 'button'		    	     ");
  theTraceWindow.document.write("    	       value   = 'Wider'>		    	     ");
  theTraceWindow.document.write("     Wider					    	     ");
  theTraceWindow.document.write("    </button>  				    	     ");
  theTraceWindow.document.write("    <button   name    = 'moreRows'		    	     ");
  theTraceWindow.document.write("    	       onclick = 'higherTraceRegion()'      	     ");
  theTraceWindow.document.write("    	       class   = 'button'		    	     ");
  theTraceWindow.document.write("    	       value   = 'Higher'>		    	     ");
  theTraceWindow.document.write("     Higher					    	     ");
  theTraceWindow.document.write("    </button>  				    	     ");
  theTraceWindow.document.write("    <button   name    = 'defaultSize'  	    	     ");
  theTraceWindow.document.write("    	       onclick = 'defaultTraceRegion()'     	     ");
  theTraceWindow.document.write("    	       class   = 'button'		    	     ");
  theTraceWindow.document.write("    	       value   = 'Default'>		    	     ");
  theTraceWindow.document.write("     Default Size				    	     ");
  theTraceWindow.document.write("    </button>  				    	     ");
  theTraceWindow.document.write("   </h1>					     	     ");
  theTraceWindow.document.write("  </form>					     	     ");
  theTraceWindow.document.write("  <textarea id      = 'traceRegion'			     ");
  theTraceWindow.document.write("  	     class   = 'traceRegion'			     ");
  theTraceWindow.document.write("  	     name    = 'abstract'			     ");
  theTraceWindow.document.write("  	     rows    = '19'				     ");
  theTraceWindow.document.write("  	     cols    = '125'></textarea> 		     ");
  theTraceWindow.document.write(" </body>   						     ");
  theTraceWindow.document.write("</html>    						     ");
  theTraceWindow.document.close();

  theTraceWindow.moveTo(   Left,    Top) ;
  theTraceWindow.resizeTo(Width, Height) ;
  
  return theTraceWindow ;
}
//=========================================================================
function DM_FormatMsg(fileCaller,functionCaller,msg)
{
 var newMsg = "["+ fileCaller + "::" + functionCaller + "()]" ;
 var len = newMsg.length ;
 var startPos = 35 - len ;
 if( startPos < 1 ) {startPos = 1 ;}
 for(var i=0; i < startPos; i++)
 {
  newMsg += " " ;
 }
 newMsg += msg;
 return newMsg ;
}
//=========================================================================
function DM_WriteWindow(theTraceWindow,msg)
{
 var theTracer	      = theTraceWindow.document.getElementById("traceRegion") ;
 var prevContent      = theTracer.innerHTML ;
 prevContent         += "\n" ;
 prevContent         += msg ;
 theTracer.innerHTML  = prevContent ;
 theTracer.scrollTop  = theTracer.scrollHeight ; // Bring last line into view
}

//=========================================================================
function DM_ClearTrace()
{
 var leftW  = top.left.theTraceWindow ;
 if( !leftW ) 
 {
  return ; // Nothing to clear
 }
 if( leftW )
 {
  theTraceWindow = leftW ;
 }
 if( rightW )
 {
  theTraceWindow = rightW ;
 }
 var theTracer	      = theTraceWindow.document.getElementById("traceRegion") ;
 theTracer.innerHTML  = "" ;
}

//---------- Additional tools under development ---------------------------s
//=========================================================================
function DM_TheWindowObject(theWindow)
{
 this.getTitle  = DM_GetTitle ;
 this.getImages = DM_GetImages ;
 this.docTitle  = theWindow.document.title ;
 this.docImages = theWindow.document.images ;
 DM_TraceWindow(this.name,theWindow.document.title) ;
}

//=========================================================================
function DM_GetTitle()
{
 return this.docTitle ;
}

//=========================================================================
function DM_GetImages()
{
 return this.docImages ;
}
