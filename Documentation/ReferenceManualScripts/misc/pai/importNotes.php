<?php
error_reporting(E_ALL);

$note_id = $_POST['note_id'];
$title = $_POST['title'];
$date = $_POST['date'];
$country_name = $_POST['country_name'];
$institute_name = $_POST['institute_name'];
$post_author = $_POST['author_fullname'];


if (sizeof($_POST) > 0){


try
  { 
    $db = new PDO('sqlite:aq.db');

    $post_author_parts = explode(" ", $post_author);

    $sql = "WHERE fullname LIKE '%".$post_author_parts[0]."%'";
    for($i=1; $i<sizeof($post_author_parts)-1; $i++){	
	$sql .= " OR fullname LIKE '%".$post_author_parts[$i]."%'";
    }

    $db_authors = $db->query("SELECT * FROM author ".$sql)->fetchAll();

    foreach($db_authors as $db_author){
	$db_author_parts = explode(" ", $db_author['fullname']);	
	
	if (sizeof(array_diff($post_author_parts, $db_author_parts)) == 0){
		$author_id = $db_author['id'];
		break;
	}
    }

    if (!isset($author_id)){
	$db->query("INSERT INTO author (id,fullname,email,cadi_url) VALUES (NULL,'".$post_author."','not_set',' ')");
    	$author_id = $db->lastInsertId();	
    }    

    $note = $db->query("SELECT * FROM note WHERE code='".$note_id."' LIMIT 1")->fetchAll();
    if (sizeof($note) == 0){
    	$db->query("INSERT INTO note (id,name,code) VALUES (NULL,'".$title."','".$note_id."')");
	$note_id = $db->lastInsertId();
    }
    else{	
	$note_id = $note[0]['id'];
    }

    $institute = $db->query("SELECT * FROM institute WHERE name='".$institute_name."' LIMIT 1")->fetchAll();
    if (sizeof($institute) == 0){
    	$db->query("INSERT INTO institute (id,name,cadi_url) VALUES (NULL,'".$institute_name."',' ')");
	$institute_id = $db->lastInsertId();
    }
    else{	
	$institute_id = $institute[0]['id'];
    }

    $country = $db->query("SELECT * FROM country WHERE name='".$country_name."' LIMIT 1")->fetchAll();
    if (sizeof($country) == 0){
    	$db->query("INSERT INTO institute (id,name,cadi_url) VALUES (NULL,'".$country_name."',' ')");
	$country_id = $db->lastInsertId();
    }
    else{	
	$country_id = $country[0]['id'];
    }

    
    $nc = $db->query("SELECT * FROM note_country WHERE note_id='".$note_id."' AND country_id=".$country_id." LIMIT 1")->fetchAll();
    if (sizeof($nc) == 0){    
	    $db->query("INSERT INTO note_country (id,note_id,country_id) VALUES (NULL,'".$note_id."','".$country_id."')");
    }

    $ni = $db->query("SELECT * FROM note_institute WHERE note_id='".$note_id."' AND institute_id=".$institute_id." LIMIT 1")->fetchAll();
    if (sizeof($ni) == 0){    
	    $db->query("INSERT INTO note_institute (id,note_id,institute_id) VALUES (NULL,'".$note_id."','".$institute_id."')");
    }

    $na = $db->query("SELECT * FROM note_author WHERE note_id='".$note_id."' AND author_id=".$author_id." LIMIT 1")->fetchAll();
    if (sizeof($na) == 0){    
	    $db->query("INSERT INTO note_author (id,note_id,author_id) VALUES (NULL,'".$note_id."','".$author_id."')");
    }

    echo "OK";

  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }
 }

 ?>