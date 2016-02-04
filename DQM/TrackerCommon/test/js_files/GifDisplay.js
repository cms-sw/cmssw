var gif_url; 

// strings containing the names of all active GifDisplays                            
var active_displays_l = new Array();

// the current displayFrame                                                          
var current_display;                                                                 
                                                                                     
// the list of displayFrame objects                                                  
var displays_l = new Array();                                                        
                                                                                     
function displayFrame(name)                                                          
{                                                                                    
  this.name = name;                                                                  
  this.is_viewed = false;                                                            
  this.viewed_l = new Array();                                                       
}                                                                                    
                                                                                     
/*                                                                                   
  This function is called onload. It creates the list of                             
  displayFrame objects.                                                              
*/                                                                                   
                                                                                     
function fillDisplayList()                                                           
{                                                                                    
  var iframes_l = document.getElementsByTagName("iframe");                           
  for (i = 0; i < iframes_l.length; i++)                                             
  {                                                                                  
    displays_l[i] = new displayFrame(iframes_l[i].id);                               
  }                                                                                  
                                                                                     
  // the default current is the first:                                               
  current_display = displays_l[0];                                                   
}                                                                                    
                                                                                     
function makeCurrent(display_frame_name)                                             
{                                                                                    
  for (i = 0; i < displays_l.length; i++)                                            
  {                                                                                  
    if (displays_l[i].name == display_frame_name)                                    
    {                                                                                
      break;                                                                         
    }                                                                                
  }                                                                                  
  current_display = displays_l[i];                                                   
}

/*                                                                                   
  Returns true if the display frame provided as an argument                          
  is currently being viewed.                                                         
*/                                                                                   
                                                                                     
function isViewed(display_frame_name)                                                
{                                                                                    
  for (i = 0; i < active_displays_l.length; i++)                                     
  {                                                                                  
    if (active_displays_l[i] == display_frame_name)                                  
    {                                                                                
      return true;                                                                   
    }                                                                                
  }                                                                                  
  return false;                                                                      
}                                                                                    
                                                                                     
//*************************************************************/                     
                                                                                     
/*                                                                                   
  These functions get called if the user clicks on the "start viewing"               
  or "stop viewing" buttons of a display frame. They set the is_viewed               
  field of the displayFrame object.                                                  
*/                                                                                   
                                                                                     
function getDisplayFrame(display_frame_name)                                         
{                                                                                    
  for (i = 0; i < displays_l.length; i++)                                            
  {                                                                                  
    if (displays_l[i].name == display_frame_name)                                    
    return displays_l[i];                                                            
  }                                                                                  
}                                                                                    
                                                                                     
function startViewing(display_frame_name)                                            
{                                                                                    
  var display = getDisplayFrame(display_frame_name);                                 
                                                                                     
  if (display.is_viewed)                                                             
  {                                                                                  
    alert('This GifViewer is already active');                                       
    return;                                                                          
  }                                                                                  
                                                                                     
  display.is_viewed = true;                                                          
  updateDisplay(display_frame_name);                                                 
}                                                                                    
 
function stopViewing(display_frame_name)                                             
{                                                                                    
  var display = getDisplayFrame(display_frame_name);                                 
  display.is_viewed = false;                                                         
}  

/*                                                                                   
  This function is initially called when the "start viewing" button                  
  of a display frame is pressed and keeps calling itself every                       
  [interval] msec, refreshing the frame until it becomes inactive.                   
*/                                                                                   
                                                                                     
function updateDisplay(display_frame_name)                                           
{                                                                                    
  var interval = 5000;                                                               
  var display_frame = getDisplayFrame(display_frame_name);                           
                                                                                     
  if (display_frame.is_viewed == true)                                               
  {                                                                                  
    makeDisplayRequest(display_frame_name);                                          
    if (display_frame.viewed_l.length != 0)                                          
    {                                                                                
      window.frames[display_frame_name].location.href = getGifURL(display_frame_name); 
    }                                                                                
  }                                                                                  
  var this_function_call = "updateDisplay('" + display_frame_name + "')";            
  setTimeout(this_function_call, interval);                                          
}                                                                                    
                                                                                     
//*************************************************************/                     
                                                                                     
function getGifURL(display_frame_name)                                               
{                                                                                    
  var url = getContextURL();                                                         
  url = url + "/temporary/" + display_frame_name + ".gif";                           
  return url;                                                                        
}                                                                                    
                                                                                     
//*************************************************************/                     
                                                                                     
function getDisplayRequestURL(display_frame_name)                                    
{                                                                                    
  url = getApplicationURL();                                                         
  url = url + "/Request"                                                             
  url = url + "?" + "RequestID=Draw"                                                 
  url = url + "&" + "Current=" + contentViewer_current;                              
  url = url + "&" + "DisplayFrameName=" + display_frame_name;                        
                                                                                     
  var display_frame = getDisplayFrame(display_frame_name);                           
  for (i = 0; i < display_frame.viewed_l.length; i++)                                
  {                                                                                  
    url = url + "&" + "View=" + display_frame.viewed_l[i];                           
  }                                                                                  
  return url;                                                                        
}

//*************************************************************/                     
                                                                                     
function makeDisplayRequest(display_frame_name)                                      
{                                                                                    
  url = getDisplayRequestURL(display_frame_name);                                    
  // pass a reference to the updateGifURL function:                                  
  makeRequest(url, updateGifURL);                                                    
}                                                                                    
                                                                                     
//*************************************************************/                     
                                                                                     
function updateGifURL()                                                              
{                                                                                    
  if (http_request.readyState == 4)                                                  
  {                                                                                  
    if (http_request.status == 200)                                                  
    {                                                                                
      var xmldoc;                                                                    
                                                                                     
       // Load the xml elements on javascript lists:                                 
      if (http_request != false)                                                     
      {                                                                              
        xmldoc  = http_request.responseXML;                                          
        gif_url = xmldoc.getElementsByTagName('fileURL').item(0).firstChild.data;    
      }                                                                              
    }                                                                                
  }                                                                                  
}            

