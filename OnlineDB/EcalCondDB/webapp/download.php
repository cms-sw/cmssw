<?php

/*
 * download.php
 *
 * Send a file to the browser, forcing a download dialogue
 * Credit given below
 *
 * $Id: download.php,v 1.2 2006/07/23 16:47:58 egeland Exp $
 */

/*
From php.net readline function page

m (at) mindplay (dot) dk
16-Jan-2005 09:20
Here's a function for sending a file to the client - it may look more
complicated than necessary, but has a number of advantages over
simpler file sending functions: - Works with large files, and uses
only an 8KB buffer per transfer.  - Stops transferring if the client
is disconnected (unlike many scripts, that continue to read and buffer
the entire file, wasting valuable resources) but does not halt the
script - Returns TRUE if transfer was completed, or FALSE if the
client was disconnected before completing the download - you'll often
need this, so you can log downloads correctly.  - Sends a number of
headers, including ones that ensure it's cached for a maximum of 2
hours on any browser/proxy, and "Content-Length" which most people
seem to forget.  (tested on Linux (Apache) and Windows (IIS5/6) under
PHP4.3.x)

Note that the folder from which protected files will be pulled, is set
as a constant in this function (/protected) ... Now here's the
function:
*/


function send_file($path) {
  ob_end_clean();
  if (preg_match(':\.\.:', $path)) { return(FALSE); }
  if (! preg_match(':^plotcache/:', $path)) { return(FALSE); }
  if (!is_file($path) or connection_status()!=0) return(FALSE);

  header("Cache-Control: no-store, no-cache, must-revalidate");
  header("Cache-Control: post-check=0, pre-check=0", false);
  header("Pragma: no-cache");
  header("Expires: ".gmdate("D, d M Y H:i:s", mktime(date("H")+2, date("i"), date("s"), date("m"), date("d"), date("Y")))." GMT");
  header("Last-Modified: ".gmdate("D, d M Y H:i:s")." GMT");
  header("Content-Type: application/octet-stream");
  header("Content-Length: ".(string)(filesize($path)));
  header("Content-Disposition: inline; filename=".basename($path));
  header("Content-Transfer-Encoding: binary\n");
  if ($file = fopen($path, 'rb')) {
    while(!feof($file) and (connection_status()==0)) {
      print(fread($file, 1024*8));
      flush();
      @ob_flush();
    }
    fclose($file);
  }
  return((connection_status()==0) and !connection_aborted());
}
?>

<?php 
if (!isset($_GET['file'])) { exit; }

if (!send_file($_GET['file'])) { 
  die ("File transfer failed"); 
// either the file transfer was incomplete 
// or the file was not found 
} else { 
// the download was a success 
// log, or do whatever else 
} 
?> 
