<?php
error_reporting(E_ALL);

$datetime = $_POST['datetime'];

$contact_url = $_POST['contact_url'];
$contact_name = $_POST['contact_name'];
$contact_email = $_POST['contact_email'];

$analysis_url = $_POST['analysis_url'];
$analysis_code = $_POST['analysis_code'];
$analysis_title = $_POST['analysis_name'];

$notes_html = $_POST['notes_html'];

$institute_name = $_POST['institute_name'];
$institute_url = $_POST['institute_url'];

$awgname = $_POST['awgname'];

//print_r($_POST);

if (sizeof($_POST) > 0){


try
  { 
    $db = new PDO('sqlite:aq.db');

    $institute = $db->query("SELECT * FROM institute WHERE name='".$institute_name."' LIMIT 1")->fetchAll();
    if (sizeof($institute) == 0){    

    	$db->query("INSERT INTO institute (id,name,cadi_url) VALUES (NULL,'".$institute_name."','".$institute_url."')");
	$institute_id = $db->lastInsertId();
    }
    else{	
	$institute_id = $institute[0]['id'];
    }

    $author = $db->query("SELECT * FROM author WHERE email='".$contact_email."' AND fullname='".$contact_name."'")->fetchAll();
    if (sizeof($author) == 0){    
    	$db->query("INSERT INTO author (id,fullname,email,cadi_url) VALUES (NULL,'".$contact_name."','".$contact_email."','".$contact_url."')");
    	$author_id = $db->lastInsertId();
    }
    else{
	$author_id = $author[0]['id'];
    }    

    $publication = $db->query("SELECT * FROM publication WHERE code='".$analysis_code."' LIMIT 1")->fetchAll();
    if (sizeof($publication) == 0){
	$parts = explode("-", $analysis_code);
	$year = "20".$parts[1];
	$db->query("INSERT INTO publication (id,title,code, date_created,cadi_url,year, notes_html) VALUES (NULL,'".$analysis_title."','".$analysis_code."','".$datetime."','".$analysis_url."', ".$year.",'".$notes_html."')");
	$publication_id = $db->lastInsertId();
    }
    else{
	$publication_id = $publication[0]['id'];
    }
    
    $pa = $db->query("SELECT * FROM publication_author WHERE author_id='".$author_id."' AND publication_id=".$publication_id." LIMIT 1")->fetchAll();
    if (sizeof($pa) == 0){    
	    $db->query("INSERT INTO publication_author (id,publication_id,author_id) VALUES (NULL,'".$publication_id."','".$author_id."')");
    }

    $ai = $db->query("SELECT * FROM author_institute WHERE author_id='".$author_id."' AND institute_id=".$institute_id." LIMIT 1")->fetchAll();
    if (sizeof($ai) == 0){    
	    $db->query("INSERT INTO author_institute (id,author_id,institute_id) VALUES (NULL,'".$author_id."','".$institute_id."')");
    }

    echo "OK";

  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
 }
if (false){
?>

<form action="?" method="POST">
$datetime 		<input type="text" name="datetime" value="$datetime"><br/>
$contact_url 		<input type="text" name="contact_url" value="$contact_url"><br/>
$contact_name 		<input type="text" name="contact_name" value="$contact_name"><br/>
$contact_email 		<input type="text" name="contact_email" value="$contact_email"><br/>
$analysis_url 		<input type="text" name="analysis_url" value="$analysis_url"><br/>
$analysis_code 		<input type="text" name="analysis_code" value="$analysis_code"><br/>
$analysis_title 	<input type="text" name="analysis_title" value="$analysis_title"><br/>
$institute_name 	<input type="text" name="institute_name" value="$institute_name"><br/>
$institute_url 		<input type="text" name="institute_url" value="$institute_url"><br/>

<input type="submit">
</form>

<?php } ?>