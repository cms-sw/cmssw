
var GifDisplay = {} ;

GifDisplay.gif_url = ""; 

// strings containing the names of all active GifDisplays                            
GifDisplay.active_displays_l = new Array();

// the current displayFrame                                                          
GifDisplay.current_display = "";                                                                 
                                                                                     
// the list of displayFrame objects                                                  
GifDisplay.displays_l = new Array();                                                        
                                                                                     
//___________________________________________________________________________________
GifDisplay.displayFrame = function(name)                                                          
{                                                                                    
  this.name = name;                                                                  
  this.is_viewed = false;                                                            
  this.viewed_l = new Array();                                                       
}                                                                                    
                                                                                                                                                                        
//___________________________________________________________________________________
/*                                                                                   
  This function is called onload. It creates the list of                             
  displayFrame objects.                                                              
*/                                                                                   
GifDisplay.fillDisplayList = function()                                                           
{                                                     // Unused?                               
  var iframes_l = document.getElementsByTagName("iframe");                           
  for (i = 0; i < iframes_l.length; i++)                                             
  {                                                                                  
    GifDisplay.displays_l[i] = new GifDisplay.displayFrame(iframes_l[i].id);                               
  }                                                                                  
                                                                                     
  // the default current is the first:                                               
  GifDisplay.current_display = GifDisplay.displays_l[0];                                                   
}                                                                                    
                                                                                     
//___________________________________________________________________________________
GifDisplay.makeCurrent = function(display_frame_name)                                             
{                                                      // Unused?                                 
  for (i = 0; i < GifDisplay.displays_l.length; i++)                                            
  {                                                                                  
    if (GifDisplay.displays_l[i].name == display_frame_name)                                    
    {                                                                                
      break;                                                                         
    }                                                                                
  }                                                                                  
  GifDisplay.current_display = GifDisplay.displays_l[i];                                                   
}

//___________________________________________________________________________________
/*                                                                                   
  Returns true if the display frame provided as an argument                          
  is currently being viewed.                                                         
*/                                                                                                                                                                       
GifDisplay.isViewed = function(display_frame_name)                                                
{                                                     // Unused?                                
  for (i = 0; i < GifDisplay.displays_l.length; i++)                                     
  {                                                                                  
    if (GifDisplay.displays_l[i] == display_frame_name)                                  
    {                                                                                
      return true;                                                                   
    }                                                                                
  }                                                                                  
  return false;                                                                      
}                                                                                    
                                                                                     
//___________________________________________________________________________________
/*                                                                                   
  These functions get called if the user clicks on the "start viewing"               
  or "stop viewing" buttons of a display frame. They set the is_viewed               
  field of the displayFrame object.                                                  
*/                                                                                   
GifDisplay.getDisplayFrame = function(display_frame_name)                                         
{                                                          // Unused?                            
  for (i = 0; i < GifDisplay.displays_l.length; i++)                                            
  {                                                                                  
    if (GifDisplay.displays_l[i].name == display_frame_name)                                    
    return GifDisplay.displays_l[i];                                                            
  }                                                                                  
}                                                                                    
                                                                                     
//___________________________________________________________________________________
GifDisplay.startViewing = function(display_frame_name)                                            
{                                                        // Unused?                            
  var display = GifDisplay.getDisplayFrame(display_frame_name);                                 
                                                                                     
  if (display.is_viewed)                                                             
  {                                                                                  
    alert('This GifViewer is already active');                                       
    return;                                                                          
  }                                                                                  
                                                                                     
  display.is_viewed = true;                                                          
  GifDisplay.updateDisplay(display_frame_name);                                                 
}                                                                                    
 
//___________________________________________________________________________________
GifDisplay.stopViewing = function(display_frame_name)                                             
{                                                         // Unused?                             
  var display = GifDisplay.getDisplayFrame(display_frame_name);                                 
  display.is_viewed = false;                                                         
}  

//___________________________________________________________________________________
/*                                                                                   
  This function is initially called when the "start viewing" button                  
  of a display frame is pressed and keeps calling itself every                       
  [interval] msec, refreshing the frame until it becomes inactive.                   
*/                                                                                   
GifDisplay.updateDisplay = function(display_frame_name)                                           
{                                                         // Unused?                            
  var interval = 5000;                                                               
  var display_frame = GifDisplay.getDisplayFrame(display_frame_name);                           
                                                                                     
  if (display_frame.is_viewed == true)                                               
  {                                                                                  
    GifDisplay.makeDisplayRequest(display_frame_name);                                          
    if (display_frame.viewed_l.length != 0)                                          
    {                                                                                
      window.frames[display_frame_name].location.href = GifDisplay.getGifURL(display_frame_name); 
    }                                                                                
  }                                                                                  
  var this_function_call = "updateDisplay('" + display_frame_name + "')";            
  setTimeout(this_function_call, interval);                                          
}                                                                                    
                                                                                     
//___________________________________________________________________________________
GifDisplay.getGifURL = function(display_frame_name)                                               
{                                                     // Unused?                                
  var url = WebLib.getContextURL();                                                         
  url = url + "/temporary/" + display_frame_name + ".gif";                           
  return url;                                                                        
}                                                                                    
                                                                                     
//___________________________________________________________________________________
GifDisplay.getDisplayRequestURL = function(display_frame_name)                                    
{                                                     // Unused?                                 
  url = WebLib.getApplicationURL();                                                         
  //url = url + "/Request?"                                                             
  url = url + "RequestID=Draw"                                                 
  url = url + "&" + "Current=" + contentViewer_current;                              
  url = url + "&" + "DisplayFrameName=" + display_frame_name;                        
                                                                                     
  var display_frame = GifDisplay.getDisplayFrame(display_frame_name);                           
  for (i = 0; i < display_frame.viewed_l.length; i++)                                
  {                                                                                  
    url = url + "&" + "View=" + display_frame.viewed_l[i];                           
  }                                                                                  
  return url;                                                                        
}

//___________________________________________________________________________________
GifDisplay.makeDisplayRequest = function(display_frame_name)                                      
{                                                       // Unused?                                   
  url = GifDisplay.getDisplayRequestURL(display_frame_name);                                    
  // pass a reference to the updateGifURL function:                                  
  WebLib.makeRequest(url, updateGifURL);                                                    
}                                                                                    
                                                                                     
//___________________________________________________________________________________
GifDisplay.updateGifURL = function()                                                              
{                                                       // Unused?                               
  if (WebLib.http_request.readyState == 4)                                                  
  {                                                                                  
    if (WebLib.http_request.status == 200)                                                  
    {                                                                                
      var xmldoc;                                                                    
                                                                                     
       // Load the xml elements on javascript lists:                                 
      if (WebLib.http_request != false)                                                     
      {                                                                              
        xmldoc  = WebLib.http_request.responseXML;                                          
        GifDisplay.gif_url = xmldoc.getElementsByTagName('fileURL').item(0).firstChild.data;    
      }                                                                              
    }                                                                                
  }                                                                                  
}    
        
var fillDisplayList = GifDisplay.fillDisplayList ; // Call to this function is generated by WebPage.cc
                                                   // which is not under our control: will try to fix this

