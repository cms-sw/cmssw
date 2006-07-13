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

function getContextURL()
{
  var app_url = getApplicationURL();
  var index = app_url.lastIndexOf("/");
  return app_url.substring(0, index);
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
  }
  if (!http_request) 
  {
    alert('Giving up :( Cannot create an XMLHTTP instance');
  }
  http_request.onreadystatechange = receiver_function;
  http_request.open('GET', url, true);
  http_request.send(null);

  return;
}

function dummy()
{
  displayMessages();
}


document.write('<script src="SERVED_DIRECTORY_URL/Navigator.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/GifDisplay.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/ContentViewer.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/ConfigBox.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/Select.js"><\/script>');
document.write('<script src="SERVED_DIRECTORY_URL/Messages.js"><\/script>');