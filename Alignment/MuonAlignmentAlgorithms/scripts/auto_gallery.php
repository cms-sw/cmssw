<!doctype html>
<html xmlns="http://www.w3.org/1999/xhtml" 
      xmlns:svg="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink">
 <head>
  <meta http-equiv="content-type" content="text/html; charset=utf-8" />
  <title>Plots</title>
  <meta name="language" content="en" />  
  <meta name="description" content="" />  
  <meta name="keywords" content="" />
  <style type="text/css">
   ul li {list-style: none; margin-bottom: 15px;}
   ul li img {display: block;}
   ul li span {display: block;}
  </style>
 </head>
 <body>
  <?php
  // open this directory 
  $myDirectory = opendir(".");

  // get each entry
  while($entryName = readdir($myDirectory)) { $dirArray[] = $entryName; }
  // close directory
  closedir($myDirectory);
  // count elements in array
  $indexCount = count($dirArray);
  sort($dirArray);
  ?>

  <ul>
   <?php
   // loop through the array of files and print them all in a list
   for($index=0; $index < $indexCount; $index++)
   {
    $extension = substr($dirArray[$index], -4);
    if ($extension == '.jpg' || $extension == '.png'){
     echo '<li><span><h2>' . $dirArray[$index] . '</h2></span><img src="' . $dirArray[$index] . '" alt="Image" 
/><hr>';
    }
   }
   // additional loop to show svgs in the end
   for($index=0; $index < $indexCount; $index++)
   {
    $extension = substr($dirArray[$index], -4);
    if ($extension == '.svg'){
     echo '<li><span><h2>' . $dirArray[$index] . '</h2></span><object data="' . $dirArray[$index] . '" type="image/svg+xml"></object><hr>
';
    }
   }
   ?>
  </ul>
</body>
</html>
