var http_request = false;                                                            
                                                                                     
var view_all_contents = true;                                                        
                                                                                     
/*
  This function should return the url of the application webpage
  without asking the server...
*/

function getApplicationURL()
{
  var url = window.location.href;
  // remove the cgi request from the end of the string
  var index = url.indexOf("?");
  if (index >= 0)
  {
    url = url.substring(0, index);
  }

  index = url.lastIndexOf("general");
  url = url.substring(0, index);

  // remove the trailing '/' from the end of the string
  index = url.lastIndexOf("/");
  if (index == url.length - 1)
  {
    url = url.substring(0, index);
  }

  return url;
}
function getApplicationURL2()
{
  var url = window.location.href;
  // remove the cgi request from the end of the string
  var index = url.indexOf("?");
  if (index >= 0)
  {
    url = url.substring(0, index);
  }

  index = url.lastIndexOf("temporary");
  url = url.substring(0, index);

  // add the cgi request
  url += "urn:xdaq-application:lid=15";
  return url;
}

function getContextURL()
{
  var app_url = getApplicationURL();
  var index = app_url.lastIndexOf("/");
  return app_url.substring(0, index);
}


function getApplicationParentURL()
{
  var url = window.opener.location.href;
  // remove the cgi request from the end of the string
  var index = url.indexOf("?");
  if (index != -1)
  {
    url = url.substring(0, index);
  }
  index = url.lastIndexOf("general");
  url = url.substring(0, index);
  // remove the trailing '/' from the end of the string
  index = url.lastIndexOf("/");
  if (index == url.length - 1)
  {
    url = url.substring(0, index);
  }
  return url;
}
/*
  This function submits a generic request in the form of a url
  and calls the receiver_function when the state of the request
  changes.
*/

function makeRequest(url, receiver_function) 
{
  http_request = false;
  if (window.XMLHttpRequest) 
  { 
    http_request = new XMLHttpRequest();
    if (http_request.overrideMimeType)
    {
      http_request.overrideMimeType('text/xml');
    }
  } else if (window.ActiveXObject) { 
    http_request = new ActiveXObject("Msxml2.XMLHTTP"); 
    if (!http_request) { 
      http_request = new ActiveXObject("Microsoft.XMLHTTP"); 
    } 
  } 

  if (http_request) { 
    initReq("GET", url, true, receiver_function); 
  }
  else { 
    alert('Giving up :( Cannot create an XMLHTTP instance');
  } 
}

function dummy()
{
  displayMessages();
}
// Initialize a request object that is already constructed 
function initReq(reqType, url, bool, respHandle) { 
  try { 
    // Specify the function that will handle the HTTP response 
    http_request.onreadystatechange = respHandle; 
    http_request.open(reqType, url, bool); 

    // if the reqType parameter is POST, then the 
    // 5th argument to the function is the POSTed data 
    if (reqType.toLowerCase() == "post") { 
      http_request.setRequestHeader("Content-Type", 
           "application/x-www-form-urlencoded; charset=UTF-8"); 
      http_request.send(arguments[4]); 
    }  
    else { 
      http_request.send(null); 
    } 
  } 
  catch (errv) { 
    alert ( 
        "The application cannot contact " + 
        "the server at the moment. " + 
        "Please try again in a few seconds.\\n" + 
        "Error detail: " + errv.message); 
  } 
}


document.write('<script src="SERVED_DIRECTORY_URL/js_files/Navigator.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/GifDisplay.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/ContentViewer.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/ConfigBox.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/Select.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/Messages.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/RequestHistos.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/CommonActions.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/RequestPlot.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/ClientActions.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/tab-view.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/context-menu.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/js_files/folder-tree-static.js"><\/script>');
