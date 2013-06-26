var ContentViewer = {} ;

ContentViewer.contentViewer_current = "top";
                                                                                     
//___________________________________________________________________________________
/*                                                                                   
  This function updates the ContentViewer "Unview" field                             
  after the user chooses to view or stop viewing something                           
*/                                                                                   
ContentViewer.updateContentViewerNoRequest = function()                                              
{                                                           // Unused?             
  var form = document.getElementById("ContentViewerForm");                           
  var view = form.View;                                                              
  var unview = form.Unview;                                                          
                                                                                     
  // first updated the list of viewed MEs                                            
  ContentViewer.updateViewedList();                                                                
                                                                                     
  // then update the Unview menu, based on the updated list:                         
  unview.options.length = 0;                                                         
  unview.options[0] = new Option("", "", true, true);                                
  var viewed_from_current = ContentViewer.getViewedFromDir(ContentViewer.contentViewer_current);                 
  for (var i = 0; i < viewed_from_current.length; i++)                               
  {                                                                                  
    unview.options[i + 1] = new Option(viewed_from_current[i], viewed_from_current[i], false, false);
  }                                                                                  
  unview.selectedIndex = 0;                                                          
                                                                                     
  // clear the lingering selection from the "View" menu                              
  view.selectedIndex = 0;                                                            
}                                                                                    
                                                                                                                                                                          
//___________________________________________________________________________________
ContentViewer.updateViewedList = function()                                                          
{                                                          // Unused?                          
  var form = document.getElementById("ContentViewerForm");                           
  var view   = form.View;                                                            
  var unview = form.Unview;                                                          
                                                                                     
  if (view.value != "")                                                              
  {                                                                                  
    var addition = view.value;                                                       
    ContentViewer.viewedListAdd(addition);                                                         
  }                                                                                  
  else if (unview.value != "")                                                       
  {                                                                                  
    var removal = unview.value;                                                      
    ContentViewer.viewedListRemove(removal);                                                       
  }                                                                                  
}                                                                                    
                                                                                     
//___________________________________________________________________________________
ContentViewer.viewedListRemove = function(removal)                                                   
{                                                         // Unused?                             
  for (i = 0; i < GifDisplay.current_display.viewed_l.length; i++)                              
  {                                                                                  
    if (removal == GifDisplay.current_display.viewed_l[i])                                      
    {                                                                                
      GifDisplay.current_display.viewed_l.splice(i, 1);                                         
    }                                                                                
  }                                                                                  
}                                                                                    
                                                                                     
//___________________________________________________________________________________
ContentViewer.viewedListAdd = function(addition)                                                   
{                           
  for (i = 0; i < GifDisplay.current_display.viewed_l.length; i++)                              
  {                                                                                  
    if (addition == GifDisplay.current_display.viewed_l[i])                                     
    {                                                                                
      return;                                                                        
    }                                                                                
  }                                                                                  
  GifDisplay.current_display.viewed_l[GifDisplay.current_display.viewed_l.length] = addition;              
}
                                                                                                                                                                         
//___________________________________________________________________________________
ContentViewer.makeContentViewerRequest = function()                                                  
{                                                         // Unused?                           
  url = ContentViewer.getContentViewerRequestURL();                                                
  WebLib.makeRequest(url, ContentViewer.updateContentViewer);                                             
}                                                                                    
                                                                                     
//___________________________________________________________________________________
ContentViewer.getContentViewerRequestURL = function()                                                
{                                                          // Unused?                           
  var form = document.getElementById("ContentViewerForm");                           
  var open = form.Open;                                                              
                                                                                     
  url = WebLib.getApplicationURL();                                                         
                                                                                     
  if (open.value != "")                                                              
  {                                                                                  
    //url = url + "/Request?";                                                          
    url = url + "RequestID=ContentsOpen";                                           
    url = url + "&" + "Current=" + ContentViewer.contentViewer_current;                            
    url = url + "&" + "Open=" + open.value;                                          
  }                                                                                  
                                                                                     
  return url;                                                                        
}                                                                                    
                                                                                     
//___________________________________________________________________________________
/*                                                                                   
  This function updates the fields of the content viewer widget                      
  after an "ContentViewerOpen" request.                                              
*/                                                                                                                                                                        
ContentViewer.updateContentViewer = function()                                                       
{                                                     // Unused?                               
  if (WebLib.http_request.readyState == 4)                                                  
  {                                                                                  
    if (WebLib.http_request.status == 200)                                                  
    {                                                                                
      var xmldoc;                                                                    
      var subdirs_l;                                                                 
      var view_l;                                                                    
      var unview_l;                                                                  
                                                                                     
      // Load the xml elements on javascript lists:                                  
      if (WebLib.http_request != false)                                                     
      {                                                                              
        xmldoc = WebLib.http_request.responseXML;                                           
                                                                                     
        // set the ContentViewer.contentViewer_current first:                                      
        ContentViewer.contentViewer_current = xmldoc.getElementsByTagName('current').item(0).firstChild.data;

        subdirs_l = xmldoc.getElementsByTagName('open');                             
        view_l = xmldoc.getElementsByTagName('view');                                
      }                                                                             
 
      // get references to the form elements so that we can update them              
      var form = document.getElementById("ContentViewerForm");                       
      var open = form.Open;                                                          
      var view = form.View;                                                          
      var unview = form.Unview;                                                      
                                                                                     
      // Update the Open menu:                                                       
      open.options.length = 0;                                                       
      open.options[0] = new Option("", "", true, true);                              
      open.options[1] = new Option("top", "top", false, false);                      
      for(var i = 0; i < subdirs_l.length; i++)                                      
      {                                                                              
        var to_open = subdirs_l.item(i).firstChild.data;                             
        open.options[i + 2] = new Option(to_open, to_open, false, false);            
      }                                                                              
      open.selectedIndex = 0;                                                        
                                                                                     
      // Update the View menu:                                                       
      view.options.length = 0;                                                       
      view.options[0] = new Option("", "", true, true);                              
      for(var i = 0; i < view_l.length; i++)                                         
      {                                                                              
        var to_view = view_l.item(i).firstChild.data;                                
        view.options[i + 1] = new Option(to_view, to_view, false, false);            
      }                                                                              
      view.selectedIndex = 0;                                                        
                                                                                     
      // Update the Unview menu:                                                     
      unview.options.length = 0;                                                     
      unview.options[0] = new Option("", "", true, true);                            
      var viewed_from_current = ContentViewer.getViewedFromDir(ContentViewer.contentViewer_current);             
      for (var i = 0; i < viewed_from_current.length; i++)                           
      {                                                                              
        unview.options[i + 1] = new Option(viewed_from_current[i], viewed_from_current[i], false, false);
      }                                                                              
      unview.selectedIndex = 0;                                                      
    }                                                                                
  }                                                                                  
}                                                                                    
 
//___________________________________________________________________________________
/*                                                                                   
  This function returns an array with all files in viewed_l that                     
  also reside in the directory dir, supplied as a parameter.                         
*/                                                                                   
ContentViewer.getViewedFromDir = function(dir)                                                       
{                                                      // Unused?                              
  var viewed_l = GifDisplay.current_display.viewed_l;                                           
  var in_dir_l = new Array();                                                        
  for (var i = 0; i < GifDisplay.current_display.viewed_l.length; i++)                          
  {                                                                                  
    var entry = viewed_l[i];                                                         
    var index = entry.lastIndexOf("/");                                              
    if (entry.substring(0, index) == dir)                                            
    {                                                                                
      in_dir_l[in_dir_l.length] = entry;                                             
    }                                                                                
  }                                                                                  
  return in_dir_l;                                                                   
}   
