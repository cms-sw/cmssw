<?php

try
  { 
    $db = new PDO('sqlite:aq.db');

    $publications = $db->query("SELECT * FROM publication")->fetchAll();
	
	foreach($publications as $publication){
		if (!is_null($publication['notes_html'])){

			$del = array('</a>', '>');

			$html = str_replace(',&nbsp;', '', $publication['notes_html']);

			$parts = explode( "the_delim", str_replace($del, "the_delim", $html) );


			for($i = 1; $i < sizeof($parts); $i+=2){
				$note_id = "CMS AN-".$parts[$i];
				

				$note = $db->query("SELECT * FROM note WHERE code='".$note_id."' LIMIT 1")->fetchAll();
				if (sizeof($note) > 0){
					echo "NOTE ID:".$note[0]['id']." - ";
					$np = $db->query("SELECT * FROM note_publication WHERE note_id='".$note[0]['id']."' AND publication_id='".$publication['id']."' LIMIT 1")->fetchAll();
					if (sizeof($np) == 0){
						$db->query("INSERT INTO note_publication (id,note_id,publication_id) VALUES (NULL,'".$note[0]['id']."','".$publication['id']."')");
						echo "ins> ";
					}
				}
				echo "publication ID: ".$publication['id']." - Note CODE: ".$note_id."<hr/>";
			}

		}
	}

  }
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>
?>